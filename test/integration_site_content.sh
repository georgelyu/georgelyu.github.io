#!/usr/bin/env bash
set -euo pipefail

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

site="${tmp_dir}/site"
JEKYLL_ENV=production bundle exec jekyll build --trace --disable-disk-cache -d "${site}" >/dev/null

python3 - "${site}" <<'PY'
import re
from pathlib import Path
from sys import argv
from xml.etree import ElementTree as ET

site = Path(argv[1])
home = site / "index.html"
publications = site / "publications/index.html"

assert home.is_file(), home
assert publications.is_file(), publications
assert (site / "404.html").is_file()

indices = {path.relative_to(site).as_posix() for path in site.rglob("index.html")}
assert indices == {"index.html", "publications/index.html"}, indices

urls = {
    element.text.strip()
    for element in ET.parse(site / "sitemap.xml").getroot().iter()
    if element.tag.endswith("loc") and element.text
}
assert urls == {
    "https://georgelyu.github.io/",
    "https://georgelyu.github.io/publications/",
}, urls

home_html = home.read_text(encoding="utf-8")
publications_html = publications.read_text(encoding="utf-8")

assert 'href="/publications/"' in home_html
assert "Chaoyang Lyu is a Postdoctoral Researcher at Shanghai AI Lab." in home_html
assert '<span class="font-weight-bold">Lyu</span>' in home_html
assert '<span class="font-weight-bold">Lyu</span>' in publications_html
assert "selected publications" not in home_html.lower()

titles = [
    "Building a Virtual Weakly-Compressible Wind Tunnel Testing Facility",
    "Fast and versatile fluid-solid coupling for turbulent flow simulation",
]
for title in titles:
    assert title in home_html, title
    assert title in publications_html, title

for relative_path in [
    "assets/img/prof_pic.jpg",
    "assets/img/favicon.svg",
    "assets/img/publication_preview/10.1145_3592394.jpg",
    "assets/img/publication_preview/10.1145_3478513_3480493.jpg",
]:
    asset = site / relative_path
    assert asset.is_file() and asset.stat().st_size > 0, asset

main_css = (site / "assets/css/main.css").read_text(encoding="utf-8").lower()
compact_main_css = main_css.replace(" ", "")
config = Path("_config.yml").read_text(encoding="utf-8")
theme_color_match = re.search(
    r'''^theme_color:\s*(?P<quote>["'])(?P<color>#[0-9a-fA-F]{6})(?P=quote)\s*(?:#.*)?$''',
    config,
    re.MULTILINE,
)
assert theme_color_match, "theme_color must be a quoted six-digit hex color in _config.yml"
theme_color = theme_color_match.group("color").lower()
red, green, blue = (int(theme_color[index : index + 2], 16) for index in (1, 3, 5))

assert f"--global-theme-color:{theme_color}" in compact_main_css
assert f"--global-hover-color:{theme_color}" in compact_main_css
assert re.search(rf"rgba\({red},{green},{blue},(?:0?\.05)\)", compact_main_css)
if theme_color != "#b509ac":
    assert "#b509ac" not in main_css
    assert "181,9,172" not in compact_main_css

assert not (site / "blog").exists()
assert not (site / "books").exists()
PY

echo "site content integration checks passed"
