from __future__ import annotations

__all__ = [
    "DateAttribute",
    "DurationAttribute",
    "RelativeDateAttribute",
    "RelativeDateDirection",
]

import dataclasses
from enum import Enum
from typing import Any

from typing_extensions import Self

from medkit.core import Attribute, dict_conv


@dataclasses.dataclass
class DateAttribute(Attribute):
    """Attribute representing an absolute date or time associated to a segment or
    entity.

    The date or time can be incomplete: each date/time component is optional but
    at least one must be provided.

    Attributes
    ----------
    uid : str
        Identifier of the attribute
    label : str
        Label of the attribute
    value : Any, optional
        String representation of the date with YYYY-MM-DD format for the date
        part and HH:MM:SS for the time part, if present. Missing components are
        replaced with question marks.
    year : int, optional
        Year component of the date
    month : int, optional
        Month component of the date
    day : int, optional
        Day component of the date
    hour : int, optional
        Hour component of the time
    minute : int, optional
        Minute component of the time
    second : int, optional
        Second component of the time
    metadata : dict of str to Any
        Metadata of the attribute
    """

    year: int | None
    month: int | None
    day: int | None
    hour: int | None
    minute: int | None
    second: int | None

    def __init__(
        self,
        label: str,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
        hour: int | None = None,
        minute: int | None = None,
        second: int | None = None,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
    ):
        value = _format_date(year, month, day, hour, minute, second)
        super().__init__(label=label, value=value, metadata=metadata, uid=uid)

        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second

    def to_brat(self) -> str:
        return self.value

    def to_spacy(self) -> str:
        return self.value

    def to_dict(self) -> dict[str, Any]:
        date_dict = {
            "uid": self.uid,
            "label": self.label,
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "hour": self.hour,
            "minute": self.minute,
            "second": self.second,
            "metadata": self.metadata,
        }
        dict_conv.add_class_name_to_data_dict(self, date_dict)
        return date_dict

    @classmethod
    def from_dict(cls, date_dict: dict[str, Any]) -> Self:
        return cls(
            uid=date_dict["uid"],
            label=date_dict["label"],
            year=date_dict["year"],
            month=date_dict["month"],
            day=date_dict["day"],
            hour=date_dict["hour"],
            minute=date_dict["minute"],
            second=date_dict["second"],
            metadata=date_dict["metadata"],
        )


@dataclasses.dataclass
class DurationAttribute(Attribute):
    """Attribute representing a time quantity associated to a segment or entity.

    Each date/time component is optional but at least one must be provided.

    Attributes
    ----------
    uid : str
        Identifier of the attribute
    label : str
        Label of the attribute
    value : Any, optional
        String representation of the duration (ex: "1 year 10 months 2 days")
    years : int
        Year component of the date quantity
    months : int
        Month component of the date quantity
    weeks : int
        Week component of the date quantity
    days : int
        Day component of the date quantity
    hours : int
        Hour component of the time quantity
    minutes : int
        Minute component of the time quantity
    seconds : int
        Second component of the time quantity
    metadata : dict of str to Any
        Metadata of the attribute
    """

    years: int
    months: int
    weeks: int
    days: int
    hours: int
    minutes: int
    seconds: int

    def __init__(
        self,
        label: str,
        years: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
    ):
        value = _format_duration(years, months, weeks, days, hours, minutes, seconds)
        super().__init__(label=label, value=value, metadata=metadata, uid=uid)

        self.years = years
        self.months = months
        self.weeks = weeks
        self.days = days
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds

    def to_brat(self) -> str:
        return self.value

    def to_spacy(self) -> str:
        return self.value

    def to_dict(self) -> dict[str, Any]:
        duration_dict = {
            "uid": self.uid,
            "label": self.label,
            "years": self.years,
            "months": self.months,
            "weeks": self.weeks,
            "days": self.days,
            "hours": self.hours,
            "minutes": self.minutes,
            "seconds": self.seconds,
            "metadata": self.metadata,
        }
        dict_conv.add_class_name_to_data_dict(self, duration_dict)
        return duration_dict

    @classmethod
    def from_dict(cls, duration_dict: dict[str, Any]) -> Self:
        return cls(
            uid=duration_dict["uid"],
            label=duration_dict["label"],
            years=duration_dict["years"],
            months=duration_dict["months"],
            weeks=duration_dict["weeks"],
            days=duration_dict["days"],
            hours=duration_dict["hours"],
            minutes=duration_dict["minutes"],
            seconds=duration_dict["seconds"],
            metadata=duration_dict["metadata"],
        )


class RelativeDateDirection(Enum):
    """Direction of a :class:`~.RelativeDateAttribute`"""

    PAST = "past"
    FUTURE = "future"


@dataclasses.dataclass
class RelativeDateAttribute(Attribute):
    """Attribute representing a relative date or time associated to a segment or
    entity, ie a date/time offset from an (unknown) reference date/time, with a
    direction.

    At least one date/time component must be non-zero.

    Attributes
    ----------
    uid : str
        Identifier of the attribute
    label : str
        Label of the attribute
    value : Any, optional
        String representation of the relative date (ex: "+ 1 year 10 months 2
        days")
    direction : RelativeDateDirection
        Direction the relative date. Ex: "2 years ago" corresponds to the `PAST`
        direction and "in 2 weeks" to the `FUTURE` direction.
    years : int
        Year component of the date offset
    months : int
        Month component of the date offset
    weeks : int
        Week component of the date offset
    days : int
        Day component of the date offset
    hours : int
        Hour component of the time offset
    minutes : int
        Minute component of the time offset
    seconds : int
        Second component of the time offset
    metadata : dict of str to Any
        Metadata of the attribute
    """

    direction: RelativeDateDirection
    years: int
    months: int
    weeks: int
    days: int
    hours: int
    minutes: int
    seconds: int

    def __init__(
        self,
        label: str,
        direction: RelativeDateDirection,
        years: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
    ):
        value = _format_relative_date(direction, years, months, weeks, days, hours, minutes, seconds)
        super().__init__(label=label, value=value, metadata=metadata, uid=uid)

        self.direction = direction
        self.years = years
        self.months = months
        self.weeks = weeks
        self.days = days
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds

    def to_brat(self) -> str:
        return self.value

    def to_spacy(self) -> str:
        return self.value

    def to_dict(self) -> dict[str, Any]:
        date_dict = {
            "uid": self.uid,
            "label": self.label,
            "direction": self.direction.value,
            "years": self.years,
            "months": self.months,
            "weeks": self.weeks,
            "days": self.days,
            "hours": self.hours,
            "minutes": self.minutes,
            "seconds": self.seconds,
            "metadata": self.metadata,
        }
        dict_conv.add_class_name_to_data_dict(self, date_dict)
        return date_dict

    @classmethod
    def from_dict(cls, date_dict: dict[str, Any]) -> Self:
        return cls(
            uid=date_dict["uid"],
            label=date_dict["label"],
            direction=RelativeDateDirection(date_dict["direction"]),
            years=date_dict["years"],
            months=date_dict["months"],
            weeks=date_dict["weeks"],
            days=date_dict["days"],
            hours=date_dict["hours"],
            minutes=date_dict["minutes"],
            seconds=date_dict["seconds"],
            metadata=date_dict["metadata"],
        )


def _format_date(
    year: int | None,
    month: int | None,
    day: int | None,
    hour: int | None,
    minute: int | None,
    second: int | None,
) -> str:
    """Return a string representation of a date with format YYYY-MM-DD for the date
    part and HH:MM:SS for the time part, if present. Missing components are
    replaced with question marks
    """
    formatted = ""
    if year is not None or month is not None or day is not None:
        if year is not None:
            formatted += f"{year:04}"
        else:
            formatted += "????"

        if month is not None:
            formatted += f"-{month:02}"
        else:
            formatted += "-??"

        if day is not None:
            formatted += f"-{day:02}"
        else:
            formatted += "-??"

    if hour is not None or minute is not None or second is not None:
        if formatted:
            formatted += " "
        if hour is not None:
            formatted += f"{hour:02}"
        else:
            formatted += "??"
        if minute is not None:
            formatted += f":{minute:02}"
        else:
            formatted += ":??"
        if second is not None:
            formatted += f"{second:02}"
        else:
            formatted += ":??"

    return formatted


def _format_duration(
    years: int | None,
    months: int | None,
    weeks: int | None,
    days: int | None,
    hours: int | None,
    minutes: int | None,
    seconds: int | None,
) -> str:
    """Return a string representation of a date/time offset.

    Ex: "1 year 10 months 2 days"
    """
    parts = []
    if years:
        parts.append(str(years) + (" year" if years == 1 else " years"))
    if months:
        parts.append(str(months) + (" month" if months == 1 else " months"))
    if weeks:
        parts.append(str(weeks) + (" week" if weeks == 1 else " weeks"))
    if days:
        parts.append(str(days) + (" day" if days == 1 else " days"))
    if hours:
        parts.append(str(hours) + (" hour" if hours == 1 else " hours"))
    if minutes:
        parts.append(str(minutes) + (" minute" if minutes == 1 else " minutes"))
    if seconds:
        parts.append(str(seconds) + (" second" if seconds == 1 else " seconds"))

    return " ".join(parts)


def _format_relative_date(
    direction: RelativeDateDirection,
    years: int | None,
    months: int | None,
    weeks: int | None,
    days: int | None,
    hours: int | None,
    minutes: int | None,
    seconds: int | None,
) -> str:
    """Return a string representation of a the date/time offset with a direction
    Ex: "+ 1 year 10 months 2 days"
    """
    prefix = "+ " if direction is RelativeDateDirection.FUTURE else "- "
    return prefix + _format_duration(years, months, weeks, days, hours, minutes, seconds)
