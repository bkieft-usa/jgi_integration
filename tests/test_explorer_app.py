"""Explorer app: builds from a synthetic bundle and the main callbacks return without errors."""
import numpy as np
import pytest

pytest.importorskip("dash")

from tests.explorer_fixtures import make_run, run_info  # noqa: E402
from tools.explorer_bundle.build import build_explorer_bundle  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    a = make_run(str(tmp_path_factory.mktemp("app")), "network_modules")
    out = build_explorer_bundle(a, explorer_config={"n_permutations": 19}, run_info=run_info("network_modules"),
                                force=True)
    from explorer.app import create_app
    return create_app(out)


def _cb(app, output_prefix):
    """The undecorated function of the callback whose outputs start with ``output_prefix``."""
    for key, val in app.callback_map.items():
        if key.lstrip(".").startswith(output_prefix):
            return val["callback"].__wrapped__
    raise KeyError(output_prefix)


FILTERS = {"fdr": 0.05, "min_group_size": 2, "include_noise": False}


def test_all_pages_render(app):
    from explorer.app import PAGES
    route = _cb(app, "page.children")
    for path in PAGES:
        assert route(path, "?group=group_1&term=kegg_module%3A%3AM00001&id=tx_g0f0") is not None


def test_search_finds_terms_groups_and_features(app):
    b = app.bundle
    assert any(o["value"] == "term:kegg_module::M00001" for o in b.search("glycolysis"))
    assert any(o["value"] == "group:group_1" for o in b.search("group_1"))
    assert any(o["value"].startswith("feature:tx_g0f0") for o in b.search("tx_g0f0"))
    assert b.search("glyco module") == b.search("module glyco")  # token order does not matter


def test_group_detail_callbacks(app):
    detail = _cb(app, "gr-summary.children")
    summary, tabs = detail("?group=group_1")
    assert tabs is not None
    comp = _cb(app, "gr-comp.figure")
    fig, table = comp("?group=group_1", "kegg_module", FILTERS, "light")
    assert len(fig.data) == 1
    within = _cb(app, "gr-within.children")
    assert within("?group=group_1", "dark") is not None
    net = _cb(app, "gr-net.figure")
    fig, summary = net("?group=group_1", ["term:kegg_module::M00001", "feature:tx_g0f0"], "group", FILTERS, "light")
    names = [t.name for t in fig.data]
    assert any(n and n.startswith("M00001") for n in names) and any(n and n.startswith("tx_g0f0") for n in names)
    assert summary is not None
    fig, summary = net("", [], "annotation", FILTERS, "dark")
    assert summary is None and len(fig.data) >= 2


def test_association_and_term_callbacks(app):
    results = _cb(app, "as-plot.figure")
    fig, table, notes = results("feature", "organism", "o0", "o1", None, "all", FILTERS, "light")
    assert len(fig.data) == 2
    b = app.bundle
    fwd = b.association_table("feature", "organism", *b.resolve_pair("feature", "organism", "o0", "o1"))
    rev = b.association_table("feature", "organism", *b.resolve_pair("feature", "organism", "o1", "o0"))
    assert np.allclose(fwd.set_index("entity_id")["effect"], -rev.set_index("entity_id")["effect"])
    assert set(rev["contrast"]) == {"o1 vs o0"}
    fig, table, notes = results("term", "lineage", "L1", "L2", "kegg_module", None, FILTERS, "dark")
    assert notes  # lineage is nested over organism
    fig, table, notes = results("group", "group", "__all__", None, None, None, FILTERS, "light")
    assert any("exactly like" in str(n) for n in notes)                 # 'group' is an alias of 'organism'
    body = _cb(app, "tm-body.children")
    assert body("?term=kegg_module%3A%3AM00001", None, FILTERS) is not None
    heat = _cb(app, "en-heat.figure")
    fig, caption = heat("kegg_module", "nlp", 1, 20, FILTERS, "light")
    assert "significant" in caption


def test_discovery_and_methods(app):
    term_body = _cb(app, "dc-term-body.children")
    for link in ("direct", "group"):
        for mode in ("any", "type"):
            assert term_body("kegg_module::M00001", mode, link, FILTERS, "light") is not None
    assert _cb(app, "dc-feature-body.children")("tx_g0f0", FILTERS) is not None
    assert _cb(app, "dc-all-body.children")("all", "all", False, FILTERS) is not None
    from explorer.pages import methods
    assert "pairwise" in str(methods.layout(app.bundle, {}))
