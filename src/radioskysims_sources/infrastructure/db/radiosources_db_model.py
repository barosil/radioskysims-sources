"""\
Módulo: src/radioskysims/infrastructure/db/radiosources_db_model.py

Modelo SQLModel para persistência de fontes de rádio (RadioSources)
no banco de dados PostgreSQL.

Este módulo pertence à camada de infraestrutura e provê a
representação persistente mínima para as entidades de domínio
``RadioSource`` e ``SpectralModel`` do contexto de RadioSources.

Unidades:
- Frequência em MHz
- Fluxo em Jy
- Ângulos em graus

Copyright BINGO Collaboration
Last modified: 2026-02-26
"""

from __future__ import annotations

from sqlmodel import Field, SQLModel

from radioskysims_sources.domain.radiosources.models import RadioSource, SpectralModel


class RadioSourceDB(SQLModel, table=True):
    """Modelo persistente para uma fonte de rádio pontual.

    Esta classe mapeia a entidade de domínio :class:`RadioSource` para
    uma tabela PostgreSQL usando SQLModel. Ela armazena os parâmetros
    mínimos necessários para reconstruir o modelo espectral associado.
    """

    __tablename__ = "radiosources"

    id: str = Field(primary_key=True, index=True)
    ra_deg: float = Field(index=True, description="Ascensão reta em graus.")
    dec_deg: float = Field(index=True, description="Declinação em graus.")
    nu0_mhz: float = Field(description="Frequência de referência em MHz.")
    s0_jy: float = Field(description="Fluxo em Jy na frequência de referência.")
    alpha: float = Field(description="Índice espectral (alpha).")

    @classmethod
    def from_domain(cls, src: RadioSource) -> RadioSourceDB:
        """Cria um :class:`RadioSourceDB` a partir de um :class:`RadioSource`."""

        return cls(
            id=src.id,
            ra_deg=src.ra_deg,
            dec_deg=src.dec_deg,
            nu0_mhz=src.spectral_model.nu0_mhz,
            s0_jy=src.spectral_model.s0_jy,
            alpha=src.spectral_model.alpha,
        )

    def to_domain(self) -> RadioSource:
        """Converte o registro persistente de volta para o domínio."""

        spectral = SpectralModel(
            nu0_mhz=self.nu0_mhz,
            s0_jy=self.s0_jy,
            alpha=self.alpha,
        )
        return RadioSource(
            id=self.id,
            ra_deg=self.ra_deg,
            dec_deg=self.dec_deg,
            spectral_model=spectral,
        )
