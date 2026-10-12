from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from radioskysims_core.config.config_loader import load_config
from radioskysims_core.paths import get_project_root
from sqlmodel import func, select

import radioskysims.resources.satellites as sats_resources
from radioskysims_sources.domain.signal.ports import SignalRepositoryPort
from radioskysims_sources.infrastructure.db.satellite.models import SatID, Signal

logger = logging.getLogger(__name__)


class SignalRepository(SignalRepositoryPort):
    def __init__(
        self,
        engine: Any = None,
        signals_dict: dict[str, Any] | None = None,
        signals_json: str | None = None,
    ) -> None:
        self.DB = engine
        self._set_source(signals_dict, signals_json)

    def _set_source(self, source_dict: dict[str, Any] | None, source_json: str | None) -> None:
        """Define a fonte dos dados de sinais GNSS.

        O caminho ``sat_signals`` é lido do **settings ativo no momento da
        construção** (e não no import do módulo), permitindo que um arquivo de
        settings nomeado aponte para um ``signals.json`` próprio (ex.: um
        experimento específico com EIRP recalculado).
        """
        self.signals_dict: dict[str, Any] = source_dict or {}
        self.signals_json: str | Path | None = source_json
        if not self.signals_dict and not self.signals_json:
            configured = (load_config(key="signals") or {}).get("sat_signals")
            configured_path = self._resolve_settings_path(configured) if configured else None
            if configured_path is not None and configured_path.exists():
                logger.info("Carregando tabela de sinais de config: %s.", configured_path)
                self.signals_json = configured_path
            else:
                logger.info("Carregando tabela de sinais padrão.")
                self.signals_json = Path(sats_resources.__path__[0]) / "sat_signals.json"
        if self.signals_json:
            self.signals_dict = self._load_signals_from_json()

    @staticmethod
    def _resolve_settings_path(value: str | Path) -> Path:
        """Ancora caminhos relativos na raiz do projeto (CWD-independente)."""
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = get_project_root() / path
        return path

    def _load_signals_from_json(self) -> dict[str, Any]:
        """Carrega sinais GNSS a partir do arquivo JSON definido em settings.toml."""
        signals_path = Path(str(self.signals_json))
        if not signals_path.exists():
            logger.error("Arquivo de sinais não encontrado: %s", signals_path)
            return {}

        try:
            with signals_path.open(encoding="utf-8") as f:
                data = json.load(f)
                # Historical formats:
                # - list[{"constellation": "...", "signals": [...]}]
                # - {"constellations": [...]} (legacy wrapper)
                if isinstance(data, list):
                    return {"constellations": data}
                if isinstance(data, dict):
                    return data
                logger.error("Formato de JSON de sinais inesperado: %s", type(data))
                return {}
        except Exception as e:
            logger.error("Erro ao carregar JSON de sinais: %s", e)
            return {}

    def _signals_from_json(self) -> None:
        """Carrega sinais GNSS a partir do arquivo JSON definido em settings.toml."""

        with self.DB.session() as session:
            for constellation_data in self.signals_dict.get("constellations", []):
                constellation = constellation_data["constellation"]

                # Normaliza o nome da constelação do JSON para o padrão usado no
                # banco (ex.: GSAT -> GALILEO, COSMOS -> GLONASS). Usa a mesma
                # lógica do get_eirp_map: `constellation_std` tem precedência.
                constellation_norm = (
                    constellation_data.get("constellation_std")
                    or self._CONSTELLATION_MAP.get(constellation)
                    or constellation
                )

                # Case-insensitive constellation match to avoid DB normalization issues.
                sats = session.exec(
                    select(SatID).where(
                        func.upper(SatID.constellation) == constellation_norm.upper()
                    )
                ).all()

                if not sats:
                    logger.warning("Nenhum satélite encontrado para %s", constellation)
                    continue

                for sig_data in constellation_data["signals"]:
                    freq_str = sig_data["frequency"].replace(" MHz", "")
                    try:
                        freq = float(freq_str)
                    except ValueError:
                        logger.error("Frequência inválida: %s", sig_data["frequency"])
                        continue

                    sig_name = sig_data["signal_name"]
                    modulation = sig_data["modulation"]

                    existing_signal = session.exec(
                        select(Signal).where(
                            Signal.signal_name == sig_name,
                            func.abs(Signal.frequency - freq) < 1e-6,
                            Signal.modulation == modulation,
                        )
                    ).first()

                    if existing_signal:
                        signal = existing_signal
                    else:
                        signal = Signal(
                            signal_name=sig_name,
                            modulation=modulation,
                            frequency=freq,
                        )
                        session.add(signal)
                        session.flush()

                    # Associação N:N
                    for sat in sats:
                        if signal not in sat.signals:
                            sat.signals.append(signal)

            session.commit()
            logger.info("Sinais carregados de %s e associados com sucesso.")

    def initialize_signals(self) -> None:
        """Inicializa sinais GNSS no banco de dados."""
        with self.DB.session() as session:
            existing_signals = session.exec(select(Signal)).first()
            if existing_signals:
                logger.info("Sinais já inicializados no banco de dados.")
                return
            logger.info("Inicializando sinais GNSS no banco de dados.")
            self._signals_from_json()

    def load_signals(self) -> list[Signal]:
        """Carrega todos os sinais GNSS do banco de dados."""
        with self.DB.session() as session:
            results: list[Signal] = list(session.exec(select(Signal)).all())
            return results

    def get_signals_by_constellation(self, constellation: str) -> list[Signal]:
        """Carrega sinais GNSS do banco de dados por constelação."""
        with self.DB.session() as session:
            results = list(
                session.exec(
                    select(Signal)
                    .join(Signal.satellites)
                    .where(SatID.constellation == constellation)
                )
                .unique()
                .all()
            )
            return results

    # ── Mapa de constelação padrão (JSON → satélite) ──
    _CONSTELLATION_MAP: dict[str, str] = {
        "GPS": "GPS",
        "GSAT": "GALILEO",
        "BEIDOU": "BEIDOU",
        "COSMOS": "GLONASS",
    }

    def get_eirp_map(self) -> dict[str, float]:
        """Retorna mapeamento constelação → EIRP (dBW).

        Usa tanto ``constellation`` (nome no JSON) quanto
        ``constellation_std`` (nome padronizado, se existir)
        como chaves para facilitar a resolução.

        Returns
        -------
        dict[str, float]
            Ex.: {"GPS": 24.6, "GALILEO": 25.8, "GLONASS": 21.1, "BEIDOU": 24.6}
        """
        eirp_map: dict[str, float] = {}
        for entry in self.signals_dict.get("constellations", []):
            eirp = entry.get("eirp_dbw")
            if eirp is None:
                continue
            # Chave pelo nome da constelação no JSON
            eirp_map[entry["constellation"]] = float(eirp)
            # Chave pelo nome padronizado (se existir)
            std = entry.get("constellation_std")
            if std:
                eirp_map[std] = float(eirp)
        return eirp_map

    def get_eirp_by_satellite(self) -> dict[int, float]:
        """Retorna EIRP **individual** por satélite (NORAD ID → dBW).

        Suporta o formato estendido de ``signals.json`` / ``sat_signals.json``,
        no qual cada satélite pode declarar sua EIRP real (medida/estimada),
        permitindo simular EIRPs individuais em vez do valor nominal da
        constelação::

            {
              "constellations": [ ... ],          # nominal por constelação
              "satellites": [
                {"norad_id": 24876, "constellation": "GPS", "eirp_dbw": 25.3},
                ...
              ]
            }

        Entradas sem ``eirp_dbw`` (ou sem ``norad_id`` válido) são ignoradas.
        O formato legado (lista pura de constelações) resulta em ``{}``.

        Returns
        -------
        dict[int, float]
            Ex.: ``{24876: 25.3, 40128: 26.1}``.
        """
        eirp_by_sat: dict[int, float] = {}
        for entry in self.signals_dict.get("satellites", []) or []:
            eirp = entry.get("eirp_dbw")
            norad_id = entry.get("norad_id")
            if eirp is None or norad_id is None:
                continue
            try:
                eirp_by_sat[int(norad_id)] = float(eirp)
            except (TypeError, ValueError):
                logger.warning("norad_id inválido na EIRP individual: %r", norad_id)
        return eirp_by_sat

    def get_signals_by_norad_ids(self, norad_ids: list[int]) -> dict[int, list[Signal]]:
        """Carrega sinais GNSS do banco de dados por NORAD ID do satélite."""
        signals: dict[int, list[Signal]] = {}
        with self.DB.session() as session:
            for norad_id in norad_ids:
                result: list[Signal] = list(
                    session.exec(
                        select(Signal).join(Signal.satellites).where(SatID.norad_id == norad_id)
                    ).all()
                )
                signals[norad_id] = result
        return signals

    def get_satellites_signals(self, sats: list[SatID]) -> list[Signal]:
        """Carrega sinais GNSS do banco de dados por NORAD ID do satélite."""
        norad_ids = [sat.norad_id for sat in sats]
        with self.DB.session() as session:
            results: list[Signal] = list(
                session.exec(
                    select(Signal).join(Signal.satellites).where(SatID.norad_id.in_(norad_ids))
                ).all()
            )
            return results

    def set_satellite_signals(self, sat: SatID, signals: list[Signal]) -> None:
        """Define os sinais GNSS associados a um satélite específico."""
        with self.DB.session() as session:
            db_sat = session.get(SatID, sat.id)
            if not db_sat:
                logger.error("Satélite não encontrado no banco de dados: %s", sat)
                return

            db_sat.signals = signals
            session.add(db_sat)
            session.commit()
            logger.info(
                "Sinais atualizados para o satélite %s (NORAD ID: %d).",
                db_sat.name,
                db_sat.norad_id,
            )
