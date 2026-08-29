# radioskysims/infrastructure/db/satellite/models.py  # noqa: I002
# Modelos para persistência de dados relacionados a satélites e sinais.

# from __future__ import annotations


from datetime import date, datetime

from sqlmodel import Field, Relationship, SQLModel


class SatSignalLink(SQLModel, table=True):  # type: ignore[call-arg, misc]
    # Link entre satélites e sinais
    __tablename__ = "sat_signal_link"
    sat_id: int | None = Field(
        default=None, foreign_key="satid.norad_id", primary_key=True
    )
    signal_id: int | None = Field(
        default=None, foreign_key="signal.id", primary_key=True
    )


class SatID(SQLModel, table=True):  # type: ignore[call-arg, misc]
    # Identificadores de satélites
    __tablename__ = "satid"
    norad_id: int = Field(primary_key=True, sa_column_kwargs={"autoincrement": False})
    name: str
    period: float | None = None
    inclination: float | None = None
    apogee: float | None = None
    perigee: float | None = None
    eccentricity: float | None = None
    gp_age: float | None = None
    constellation: str | None = None
    type: str | None = Field(default="GNSS")

    # Relacionamentos
    elements: list["OrbitalElement"] = Relationship(back_populates="satellite")
    signals: list["Signal"] = Relationship(
        back_populates="satellites",
        link_model=SatSignalLink,
    )


class OrbitalElement(SQLModel, table=True):  # type: ignore[call-arg, misc]
    __tablename__ = "orbital_elements"

    id: int | None = Field(default=None, primary_key=True)

    # Metadados
    ccsds_omm_vers: str | None = None
    comment: str | None = None
    creation_date: datetime | None = None
    originator: str | None = None

    # Identificadores
    object_name: str | None = None
    object_id: str | None = None
    norad_id: int | None = Field(default=None, foreign_key="satid.norad_id", index=True)

    # Referenciais
    center_name: str | None = None
    ref_frame: str | None = None
    time_system: str | None = None
    mean_element_theory: str | None = None

    # Elementos orbitais
    epoch: datetime | None = None
    mean_motion: float | None = None
    eccentricity: float | None = None
    inclination: float | None = None
    ra_of_asc_node: float | None = None
    arg_of_pericenter: float | None = None
    mean_anomaly: float | None = None

    # Extras
    ephemeris_type: int | None = None
    classification_type: str | None = None
    element_set_no: int | None = None
    rev_at_epoch: int | None = None
    bstar: float | None = None
    mean_motion_dot: float | None = None
    mean_motion_ddot: float | None = None
    semimajor_axis: float | None = None
    period: float | None = None
    apoapsis: float | None = None
    periapsis: float | None = None

    object_type: str | None = None
    rcs_size: str | None = None
    country_code: str | None = None
    launch_date: date | None = None
    site: str | None = None
    decay_date: str | None = None

    file: int | None = None
    gp_id: int | None = None

    # Linhas TLE
    tle_line0: str | None = None
    tle_line1: str | None = None
    tle_line2: str | None = None

    # Relacionamento N:1
    satellite: SatID | None = Relationship(back_populates="elements")


class Signal(SQLModel, table=True):  # type: ignore[call-arg, misc]
    __tablename__ = "signal"

    id: int = Field(primary_key=True)
    signal_name: str
    modulation: str
    frequency: float  # MHz
    satellites: list[SatID] = Relationship(
        back_populates="signals",
        link_model=SatSignalLink,
    )
