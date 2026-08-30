from __future__ import annotations

from dataclasses import dataclass
from typing import Final

_GALACTIC_COMPONENTS: Final[frozenset[str]] = frozenset(
    {
        "synchrotron",
        "dust",
        "freefree",
    }
)


@dataclass(frozen=True, slots=True)
class ContinuumSource:
    """
    Fonte astrofísica difusa representada por mapas do céu.

    Diferentemente de SignalSource, uma ContinuumSource não é descrita
    por uma posição/trajetória única nem por uma PSD pontual.

    Sua contribuição para o TOD é obtida através de mapas HEALPix
    avaliados na direção de apontamento do radiotelescópio.
    """

    name: str
    components: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            msg = "ContinuumSource name cannot be empty."
            raise ValueError(msg)

        if not self.components:
            msg = "ContinuumSource must define at least one physical component."
            raise ValueError(msg)

        if not all(isinstance(component, str) for component in self.components):
            msg = "ContinuumSource components must be strings."
            raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class Galaxy(ContinuumSource):
    """
    Emissão difusa da Via Láctea.

    O modelo físico inicial utiliza os componentes galácticos suportados
    pelo PySM3 solicitados para a simulação:

    - synchrotron;
    - thermal dust;
    - free-free.

    Os mapas correspondentes são gerados posteriormente pela camada de
    Application através de SkyMapGeneratorPort.
    """

    name: str = "Milky Way"
    components: tuple[str, ...] = (
        "synchrotron",
        "dust",
        "freefree",
    )

    def __post_init__(self) -> None:
        super(Galaxy, self).__post_init__()

        unsupported = set(self.components) - _GALACTIC_COMPONENTS

        if unsupported:
            msg = (
                "Unsupported Galaxy components: "
                f"{sorted(unsupported)}. "
                f"Supported components: {sorted(_GALACTIC_COMPONENTS)}."
            )
            raise ValueError(
                msg
            )


@dataclass(frozen=True, slots=True)
class NeutralHydrogen(ContinuumSource):
    """
    Emissão cosmológica difusa da linha de 21 cm do hidrogênio neutro.

    A fonte representa o sinal de HI pós-reionização utilizado em
    intensity mapping.

    A construção efetiva dos mapas depende de:

    - cosmologia;
    - espectro de potência da matéria;
    - distribuição de HI;
    - bias do HI;
    - temperatura média de brilho;
    - realização estatística do campo de densidade.

    Esses cálculos pertencem ao domínio cosmológico e aos respectivos
    adapters científicos, não a esta entidade.
    """

    name: str = "Neutral Hydrogen"
    components: tuple[str, ...] = ("21cm",)

    omega_hi_0: float = 4.0e-4
    hi_bias_0: float = 1.0

    def __post_init__(self) -> None:
        super(NeutralHydrogen, self).__post_init__()

        if self.components != ("21cm",):
            msg = "NeutralHydrogen currently supports only the '21cm' component."
            raise ValueError(
                msg
            )

        if self.omega_hi_0 <= 0:
            msg = "omega_hi_0 must be positive."
            raise ValueError(msg)

        if self.hi_bias_0 <= 0:
            msg = "hi_bias_0 must be positive."
            raise ValueError(msg)
