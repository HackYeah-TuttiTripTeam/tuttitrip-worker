"""Render the synthetic receipts of ``tests/golden/read_receipt`` (never in CI).

Every merchant, address, tax number and card number is invented, so the golden
set holds no personal data. The script writes the PNG images and
``examples.jsonl`` (reference answers) next to each other; rerun it to change
the set. The look depends on the font, so the images are committed.

    uv run python scripts/make_golden_receipts.py [--font PATH]
"""

import argparse
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1] / "tests/golden/read_receipt"
DEFAULT_FONT = "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf"
FONT_SIZE = 20
LINE_HEIGHT = 26
MARGIN = 24
RECEIPT_COLUMNS = 38
RECEIPT_BG = (250, 249, 244)
INK = (28, 28, 28)
GRAIN_MOD = 97
GRAIN_X = 7
GRAIN_Y = 13
GRAIN_DIVISOR = 40
GRAIN_MIN = 90
GRAIN_SPAN = 110
PL = "pl"
RECEIPT = "receipt"
NOISE_RADIUS = 0.8
BANK_WIDTH = 420
BANK_BG = (245, 247, 250)
BANK_CARD = (255, 255, 255)
BANK_MUTED = (110, 118, 130)
BANK_AMOUNT_SIZE = 40
PL_MONTHS = [
    "stycznia",
    "lutego",
    "marca",
    "kwietnia",
    "maja",
    "czerwca",
    "lipca",
    "sierpnia",
    "września",
    "października",
    "listopada",
    "grudnia",
]
EN_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


@dataclass(frozen=True)
class Slip:
    """One receipt or bank screenshot and what it shows."""

    id: str
    lang: Literal["pl", "en"]
    kind: Literal["receipt", "bank"]
    merchant: str
    day: date
    currency: str
    lines: tuple[tuple[str, int], ...] = ()
    total: int | None = None
    """Printed total; ``None`` = the total is cropped out and not visible."""
    sign: str = "zł"
    """Currency as printed (zł, PLN, €, $ ...)."""
    address: str = ""
    time: str = "13:42"
    date_style: Literal["dots", "iso", "long", "short"] = "dots"
    rotate: float = 0.0
    noisy: bool = False
    tags: tuple[str, ...] = ()
    notes: str = ""
    extra: tuple[tuple[str, int], ...] = field(default=())
    """Extra printed lines such as a discount (negative)."""


def money(minor: int, lang: str) -> str:
    """Format minor units the way the language writes them.

    Args:
        minor: Amount in minor units.
        lang: ``pl`` writes ``12,50``, ``en`` writes ``12.50``.

    Returns:
        The amount as text.
    """
    sign = "-" if minor < 0 else ""
    whole, cents = divmod(abs(minor), 100)
    separator = "," if lang == PL else "."
    return f"{sign}{whole}{separator}{cents:02d}"


def written_date(slip: Slip) -> str:
    """The date as printed.

    Args:
        slip: The slip.

    Returns:
        Text such as ``14.08.2026`` or ``14 sierpnia 2026``.
    """
    d = slip.day
    match slip.date_style:
        case "iso":
            return d.isoformat()
        case "short":
            return f"{d.day:02d}.{d.month:02d}.{d.year % 100:02d}"
        case "long":
            months = PL_MONTHS if slip.lang == PL else EN_MONTHS
            if slip.lang == PL:
                return f"{d.day} {months[d.month - 1]} {d.year}"
            return f"{months[d.month - 1]} {d.day}, {d.year}"
        case _:
            return f"{d.day:02d}.{d.month:02d}.{d.year}"


def printed_total(slip: Slip) -> int:
    """Total of the lines, extra lines included.

    Args:
        slip: The slip.

    Returns:
        Minor units.
    """
    return sum(amount for _, amount in slip.lines) + sum(a for _, a in slip.extra)


def render_receipt(slip: Slip, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Draw a paper receipt.

    Args:
        slip: The slip.
        font: Monospace font.

    Returns:
        The image.
    """
    pl = slip.lang == PL
    rows: list[str] = [slip.merchant.center(RECEIPT_COLUMNS)]
    if slip.address:
        rows.append(slip.address.center(RECEIPT_COLUMNS))
    rows += [
        "NIP 111-111-11-11".center(RECEIPT_COLUMNS),
        ("PARAGON FISKALNY" if pl else "RECEIPT").center(RECEIPT_COLUMNS),
        f"{written_date(slip)}  {slip.time}".center(RECEIPT_COLUMNS),
        "-" * RECEIPT_COLUMNS,
    ]
    for name, amount in (*slip.lines, *slip.extra):
        price = money(amount, slip.lang)
        room = RECEIPT_COLUMNS - len(price)
        rows.append(f"{name[: room - 1]:<{room}}{price}")
    rows.append("-" * RECEIPT_COLUMNS)
    if slip.total is not None:
        label = "SUMA" if pl else "TOTAL"
        value = f"{money(slip.total, slip.lang)} {slip.sign}"
        rows.append(f"{label:<{RECEIPT_COLUMNS - len(value)}}{value}")
        rows.append(
            ("Karta płatnicza" if pl else "Paid by card").center(RECEIPT_COLUMNS)
        )
        rows.append(
            ("Dziękujemy za zakupy" if pl else "Thank you").center(RECEIPT_COLUMNS)
        )
    height = 2 * MARGIN + LINE_HEIGHT * len(rows)
    width = 2 * MARGIN + int(font.getlength("M" * RECEIPT_COLUMNS))
    image = Image.new("RGB", (width, height), RECEIPT_BG)
    draw = ImageDraw.Draw(image)
    for number, row in enumerate(rows):
        draw.text((MARGIN, MARGIN + number * LINE_HEIGHT), row, font=font, fill=INK)
    return image


def render_bank(
    slip: Slip, font: ImageFont.FreeTypeFont, big: ImageFont.FreeTypeFont
) -> Image.Image:
    """Draw a banking app screen with one card payment.

    Args:
        slip: The slip.
        font: Regular font.
        big: Large font for the amount.

    Returns:
        The image.
    """
    pl = slip.lang == PL
    amount = slip.total if slip.total is not None else printed_total(slip)
    image = Image.new("RGB", (BANK_WIDTH, 520), BANK_BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((16, 16, BANK_WIDTH - 16, 504), radius=18, fill=BANK_CARD)
    draw.text(
        (36, 40), "Płatność kartą" if pl else "Card payment", font=font, fill=BANK_MUTED
    )
    draw.text((36, 90), f"-{money(amount, slip.lang)} {slip.sign}", font=big, fill=INK)
    draw.text((36, 160), slip.merchant, font=font, fill=INK)
    draw.text(
        (36, 200), f"{written_date(slip)}, {slip.time}", font=font, fill=BANK_MUTED
    )
    draw.text(
        (36, 260),
        ("Karta •••• 4821" if pl else "Card •••• 4821"),
        font=font,
        fill=BANK_MUTED,
    )
    draw.text(
        (36, 300),
        ("Status: zaksięgowana" if pl else "Status: posted"),
        font=font,
        fill=BANK_MUTED,
    )
    draw.text(
        (36, 340),
        ("Kategoria: Płatności kartą" if pl else "Category: Card payments"),
        font=font,
        fill=BANK_MUTED,
    )
    return image


def expected(slip: Slip) -> dict[str, object]:
    """Reference answer for a slip.

    Args:
        slip: The slip.

    Returns:
        The fields the scorer compares.
    """
    shown = (*slip.lines, *slip.extra) if slip.kind == RECEIPT else ()
    total = slip.total
    return {
        "amount_minor": total,
        "currency": slip.currency,
        "spent_on": slip.day.isoformat(),
        "merchant": slip.merchant,
        "items": [{"name": name, "amount_minor": amount} for name, amount in shown],
    }


def slips() -> list[Slip]:
    """The whole set.

    Returns:
        All fictional slips, Polish first.
    """
    d = date
    r, b = "receipt", "bank"
    return [
        Slip(
            "pl-01",
            "pl",
            r,
            "Delikatesy Pod Kasztanem",
            d(2026, 8, 14),
            "PLN",
            (
                ("Chleb wiejski", 649),
                ("Mleko 3,2% 1l", 319),
                ("Ser żółty 250g", 1299),
                ("Jabłka szara reneta", 1087),
            ),
            total=3354,
            address="ul. Długa 12, Kraków",
            tags=("easy",),
        ),
        Slip(
            "pl-02",
            "pl",
            r,
            "Restauracja Pod Lipą",
            d(2026, 8, 15),
            "PLN",
            (
                ("Żurek w chlebie", 2400),
                ("Pierogi ruskie", 2800),
                ("Kompot", 900),
                ("Szarlotka", 1600),
            ),
            total=7700,
            address="Rynek 4, Sandomierz",
            time="14:07",
            tags=("restaurant",),
        ),
        Slip(
            "pl-03",
            "pl",
            r,
            "Stacja Paliw Meteor",
            d(2026, 8, 16),
            "PLN",
            (("Benzyna 95", 28950),),
            total=28950,
            address="ul. Trasa Zakopiańska 100",
            time="08:15",
            tags=("fuel",),
        ),
        Slip(
            "pl-04",
            "pl",
            r,
            "Apteka Zdrowie",
            d(2026, 8, 17),
            "PLN",
            (
                ("Plastry opatrunkowe", 1290),
                ("Krem z filtrem SPF50", 4590),
                ("Woda termalna", 2490),
            ),
            total=8370,
            address="ul. Floriańska 3, Kraków",
            tags=("pharmacy",),
        ),
        Slip(
            "pl-05",
            "pl",
            r,
            "Kawiarnia Cynamon",
            d(2026, 8, 18),
            "PLN",
            (("Latte", 1400), ("Flat white", 1300), ("Sernik", 1800)),
            total=4500,
            date_style="iso",
            time="10:31",
            tags=("iso-date",),
        ),
        Slip(
            "pl-06",
            "pl",
            r,
            "Piekarnia Złoty Kłos",
            d(2026, 8, 19),
            "PLN",
            (("Bułka kajzerka x6", 540), ("Chleb razowy", 790), ("Drożdżówka", 480)),
            total=1810,
            date_style="short",
            time="07:40",
            tags=("short-date",),
            notes="Data 19.08.26 oznacza 2026-08-19.",
        ),
        Slip(
            "pl-07",
            "pl",
            r,
            "Delikatesy Pod Kasztanem",
            d(2026, 8, 20),
            "PLN",
            (("Makaron penne", 599), ("Sos pomidorowy", 849), ("Parmezan 100g", 1599)),
            total=2547,
            extra=(("RABAT", -500),),
            address="ul. Długa 12, Kraków",
            tags=("discount",),
            notes="Rabat to drukowana linia z kwotą ujemną; suma uwzględnia rabat.",
        ),
        Slip(
            "pl-08",
            "pl",
            r,
            "Księgarnia Kartka",
            d(2026, 8, 21),
            "PLN",
            (("Przewodnik Tatry", 4990), ("Mapa Zakopane 1:25000", 1990)),
            total=6980,
            rotate=1.2,
            tags=("rotated",),
            address="Krupówki 20, Zakopane",
        ),
        Slip(
            "pl-09",
            "pl",
            r,
            "Kiosk Ruchu 24",
            d(2026, 8, 22),
            "PLN",
            (("Bilet autobusowy x3", 1500), ("Woda Żywiec 0,5l", 350)),
            total=1850,
            noisy=True,
            tags=("noisy",),
            time="09:12",
        ),
        Slip(
            "pl-10",
            "pl",
            r,
            "Bar Mleczny Smaczek",
            d(2026, 8, 23),
            "PLN",
            (
                ("Rosół z makaronem", 650),
                ("Kotlet schabowy", 1950),
                ("Ziemniaki", 450),
                ("Surówka", 400),
            ),
            total=3450,
            sign="PLN",
            tags=("pln-sign",),
            address="ul. Mickiewicza 8",
        ),
        Slip(
            "pl-11",
            "pl",
            r,
            "Sklep Wielobranżowy Jaś",
            d(2026, 8, 24),
            "PLN",
            (
                ("Krem do opalania", 3499),
                ("Okulary przeciwsłoneczne", 4999),
                ("Czapka z daszkiem", 2999),
                ("Ręcznik plażowy", 5999),
            ),
            total=17496,
            rotate=-1.5,
            noisy=True,
            tags=("long", "noisy"),
        ),
        Slip(
            "pl-12",
            "pl",
            r,
            "Piekarnia Złoty Kłos",
            d(2026, 8, 25),
            "PLN",
            (("Chleb żytni", 790), ("Rogal", 350), ("Pączek", 450)),
            total=None,
            tags=("cropped",),
            notes="Dół paragonu ucięty: sumy nie widać, więc kwota to null.",
        ),
        Slip(
            "pl-13",
            "pl",
            r,
            "Stacja Paliw Meteor",
            d(2026, 8, 26),
            "PLN",
            (("Olej napędowy", 41210), ("Myjnia żeton", 1500)),
            total=42710,
            address="ul. Zakopiańska 5",
            time="17:55",
            tags=("fuel",),
        ),
        Slip(
            "pl-14",
            "pl",
            r,
            "Pizzeria Margherita",
            d(2026, 8, 27),
            "PLN",
            (
                ("Pizza Capricciosa", 3200),
                ("Pizza Diavola", 3400),
                ("Cola 0,5l x2", 1400),
                ("Sałatka grecka", 2200),
            ),
            total=10200,
            date_style="long",
            tags=("long-date",),
            notes="Data napisana słownie: 27 sierpnia 2026.",
        ),
        Slip(
            "pl-15",
            "pl",
            b,
            "Cukiernia Wawel",
            d(2026, 8, 14),
            "PLN",
            total=3400,
            time="12:15",
            tags=("bank",),
        ),
        Slip(
            "pl-16",
            "pl",
            b,
            "Parking Centrum Gdańsk",
            d(2026, 8, 15),
            "PLN",
            total=2500,
            date_style="long",
            time="09:03",
            tags=("bank", "long-date"),
        ),
        Slip(
            "pl-17",
            "pl",
            b,
            "Bilety MPK Kraków",
            d(2026, 8, 16),
            "PLN",
            total=1500,
            sign="PLN",
            tags=("bank",),
        ),
        Slip(
            "pl-18",
            "pl",
            b,
            "Hotel Górski Zakopane",
            d(2026, 8, 17),
            "PLN",
            total=125000,
            date_style="iso",
            tags=("bank", "large"),
        ),
        Slip(
            "pl-19",
            "pl",
            b,
            "Kantor Wymiany Walut",
            d(2026, 8, 18),
            "EUR",
            total=20000,
            sign="EUR",
            tags=("bank", "foreign"),
        ),
        Slip(
            "pl-20",
            "pl",
            r,
            "Kavárna U Mostu",
            d(2026, 8, 19),
            "CZK",
            (("Káva", 6500), ("Dort", 9500), ("Voda", 4500)),
            total=20500,
            sign="Kč",
            address="Karlova 10, Praha",
            tags=("foreign", "czk"),
        ),
        Slip(
            "en-01",
            "en",
            r,
            "Harbor Coffee Co.",
            d(2026, 7, 4),
            "USD",
            (
                ("Cold brew", 450),
                ("Blueberry muffin", 395),
                ("Bagel with cream cheese", 525),
            ),
            total=1370,
            sign="$",
            address="12 Pier Street",
            tags=("easy",),
        ),
        Slip(
            "en-02",
            "en",
            r,
            "The Old Pier Fish & Chips",
            d(2026, 7, 5),
            "GBP",
            (("Cod and chips", 1195), ("Mushy peas", 250), ("Pot of tea", 220)),
            total=1665,
            sign="£",
            date_style="long",
            tags=("long-date",),
        ),
        Slip(
            "en-03",
            "en",
            r,
            "Café Lumière",
            d(2026, 7, 6),
            "EUR",
            (("Croissant x2", 560), ("Café crème", 380), ("Jus d'orange", 450)),
            total=1390,
            sign="€",
            rotate=1.0,
            noisy=True,
            tags=("foreign", "noisy"),
        ),
        Slip(
            "en-04",
            "en",
            b,
            "Bluebird Taxi",
            d(2026, 7, 7),
            "GBP",
            total=2340,
            sign="£",
            date_style="long",
            tags=("bank",),
        ),
        Slip(
            "en-05",
            "en",
            b,
            "Riverside Tickets",
            d(2026, 7, 8),
            "USD",
            total=6000,
            sign="$",
            tags=("bank",),
        ),
    ]


def add_grain(image: Image.Image) -> Image.Image:
    """Speckle an image with a fixed pattern (a worn, photographed look).

    Args:
        image: The clean image.

    Returns:
        The same image with grey specks.
    """
    pixels = image.load()
    if pixels is None:
        return image
    for x in range(image.width):
        for y in range(image.height):
            if (x * GRAIN_X + y * GRAIN_Y) % GRAIN_DIVISOR == 0:
                grain = GRAIN_MIN + (x * y) % GRAIN_SPAN
                pixels[x, y] = (grain, grain, grain)
    return image


def main() -> None:
    """Write the images and ``examples.jsonl``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", default=DEFAULT_FONT)
    args = parser.parse_args()
    font = ImageFont.truetype(args.font, FONT_SIZE)
    big = ImageFont.truetype(args.font, BANK_AMOUNT_SIZE)
    images = ROOT / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for slip in slips():
        # The printed total must match the lines unless the slip says otherwise.
        if slip.kind == RECEIPT and slip.total not in {None, printed_total(slip)}:
            msg = f"{slip.id}: the printed total does not match the lines"
            raise ValueError(msg)
        image = (
            render_receipt(slip, font)
            if slip.kind == RECEIPT
            else render_bank(slip, font, big)
        )
        if slip.rotate:
            image = image.rotate(slip.rotate, expand=True, fillcolor=RECEIPT_BG)
        if slip.noisy:
            image = add_grain(image.filter(ImageFilter.GaussianBlur(NOISE_RADIUS)))
        name = f"{slip.id}.png"
        image.save(images / name, optimize=True)
        rows.append(
            {
                "id": slip.id,
                "lang": slip.lang,
                "tags": [slip.kind, *slip.tags],
                "input": {"image": f"images/{name}"},
                "expected": expected(slip),
                "notes": slip.notes,
            }
        )
    with (ROOT / "examples.jsonl").open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    print(f"{len(rows)} slips in {ROOT}")


if __name__ == "__main__":
    main()
