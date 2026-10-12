from __future__ import annotations

import shutil
import json
import subprocess
import unittest
from pathlib import Path

import citation_graph.server as server


class StaticAssetTests(unittest.TestCase):
    """The graph UI's CSS/JS live in citation_graph/static/, not in server.py.

    They used to sit inside Python string literals in _build_sigma_html, a
    4,432-line function, where no syntax checker, linter or editor could see
    them. A ReferenceError shipped that way on 2026-08-03: esc() was defined
    inside a fetch().then() callback and called from the sibling .catch(), so
    every failed graph load replaced the real error message with a crash.
    Nothing could catch it before the browser did.

    These tests keep the files present, served correctly, and referenced by
    the generated shell.
    """

    def setUp(self):
        self.static_dir = server._STATIC_DIR

    def _html(self) -> str:
        return server._build_sigma_html(
            n_items=0, n_nodes=0, n_edges=0, n_citer=0, n_ref=0,
            palette=server._PALETTE, css_root=server._CSS_ROOT,
            js_theme=server._JS_THEME,
        )

    def test_both_assets_exist_and_are_not_empty(self):
        # Guards a .gitignore or packaging miss: without the files the page
        # loads and then hangs on the spinner with one console 404.
        for name in ("app.css", "app.js"):
            path = self.static_dir / name
            self.assertTrue(path.is_file(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 1000, name)

    def test_each_asset_is_served_with_an_explicit_content_type(self):
        # Starlette falls back to text/plain when mimetypes misses, and a
        # text/plain stylesheet is rejected outright by browsers in standards
        # mode -- the whole page would render unstyled. The media_type must
        # not depend on the host's mimetypes database.
        for name, expected in (("app.css", "text/css"), ("app.js", "text/javascript")):
            response = server._route_static_asset(name)
            self.assertEqual(response.status_code, 200, name)
            self.assertTrue(
                response.media_type.startswith(expected),
                f"{name}: {response.media_type!r} does not start with {expected!r}",
            )
            self.assertEqual(Path(response.path).name, name)

    def test_unknown_names_are_rejected(self):
        for name in ("../server.py", "nope.js", "app.js.bak", ""):
            response = server._route_static_asset(name)
            self.assertEqual(response.status_code, 404, name)

    def test_assets_are_revalidated_on_every_request(self):
        # _route_index returns the *previous* build's HTML and rebuilds in the
        # background, so a ?v= cache buster baked into the shell would always
        # be one page load stale -- reintroducing the "reload twice to see
        # your edit" confusion. A response-time no-cache header cannot go
        # stale that way.
        for name in ("app.css", "app.js"):
            response = server._route_static_asset(name)
            self.assertEqual(response.headers["cache-control"], "no-cache", name)

    def test_shell_links_both_assets_and_injects_the_palette(self):
        html = self._html()
        self.assertIn('<link rel="stylesheet" href="/static/app.css">', html)
        self.assertIn('<script src="/static/app.js"></script>', html)
        self.assertIn("window.__RAG_THEME__", html)
        self.assertIn(server._PALETTE["nodeZotero"], html)

    def test_admin_visual_tokens_match_the_graph_palette(self):
        css = (self.static_dir / "admin.css").read_text(encoding="utf-8")
        for key in (
            "nodeZotero", "nodeCiter", "nodeRef", "surface",
            "surfaceContainerLow", "surfaceContainerHigh", "outlineVariant",
            "onSurface", "onSurfaceVariant", "textDis",
        ):
            declaration = f"{server._palette_css_var(key)}: {server._PALETTE[key]};"
            self.assertIn(declaration, css, key)
        self.assertNotIn("Georgia", css)

    def test_admin_skips_the_confirmation_dialog_for_jobs_without_a_phrase(self):
        script = (self.static_dir / "admin.js").read_text(encoding="utf-8")
        self.assertIn("if(!def.paid) { void startDefinition(def); return; }", script)
        self.assertIn('$("confirm-form").addEventListener("submit"', script)
        self.assertIn('$("confirm-dialog").close("confirm")', script)

    def test_admin_shell_versions_both_browser_assets(self):
        html = (self.static_dir / "admin.html").read_text(encoding="utf-8")
        self.assertIn('/admin/assets/admin.css?v=', html)
        self.assertIn('/admin/assets/admin.js?v=', html)

    def test_admin_has_a_current_audit_diagnostics_region(self):
        html = (self.static_dir / "admin.html").read_text(encoding="utf-8")
        script = (self.static_dir / "admin.js").read_text(encoding="utf-8")
        self.assertIn('id="audit-diagnostics" aria-live="polite"', html)
        self.assertIn('$("audit-diagnostics")', script)
        self.assertIn("failure.message", script)
        self.assertIn("failure.action", script)
        self.assertIn("gate.available", script)

    def test_admin_explains_missing_sources_in_old_and_new_failure_logs(self):
        if shutil.which("node") is None:
            self.skipTest("node is not installed")
        script = (self.static_dir / "admin.js").read_text(encoding="utf-8")
        functions = script[script.index("function legacyMissingFileMessages"):
                           script.index("async function showFailure")]
        old_log = ("Redirect location: 'file:///library/storage/ATT12345/Example%20Book.pdf'\n"
                   "RuntimeError: Eligible attachment ATT12345 has no readable local file\n")
        cases = [old_log, old_log.replace("Example%20Book.pdf", "%INVALID"),
                 "[停止] Example Book\n[必要な作業] Zoteroで添付を開いてください。\n",
                 "unrelated error\n"]
        result = subprocess.run(
            ["node", "-e", functions +
             "console.log(JSON.stringify(JSON.parse(process.argv[1]).map(failureMessages)));",
             json.dumps(cases)], capture_output=True, text=True, check=True,
        )
        messages = json.loads(result.stdout)
        self.assertIn("Example Book.pdf", messages[0][0])
        self.assertIn("ATT12345", messages[0][0])
        self.assertIn("ダウンロード・同期", messages[0][1])
        self.assertIn("再実行", messages[0][1])
        self.assertIn("ATT12345", messages[1][0])
        self.assertEqual(messages[2], ["[停止] Example Book", "[必要な作業] Zoteroで添付を開いてください。"])
        self.assertEqual(messages[3], [])

    def test_admin_shows_remote_mistral_batch_progress(self):
        script = (self.static_dir / "admin.js").read_text(encoding="utf-8")
        self.assertIn('metric("Mistral Batch"', script)
        self.assertIn("batch.completed_requests", script)

    def test_admin_shows_accessible_estimated_job_progress(self):
        script = (self.static_dir / "admin.js").read_text(encoding="utf-8")
        css = (self.static_dir / "admin.css").read_text(encoding="utf-8")
        self.assertIn("function progressEstimate(job,log=\"\")", script)
        self.assertIn('track.setAttribute("role","progressbar")', script)
        self.assertIn("経過時間からの目安", script)
        self.assertIn("ログ実測", script)
        self.assertIn(".progress-track.estimated", css)
        self.assertIn("prefers-reduced-motion", css)

    def test_app_js_stays_a_classic_synchronous_script(self):
        # The JS has no DOMContentLoaded guard and touches
        # getElementById('loading') on its first statement, and it depends on
        # graphology/sigma/d3 already being evaluated. defer, async, or
        # type="module" would all break that ordering.
        html = self._html()
        tag = html[html.index('<script src="/static/app.js"'):]
        tag = tag[:tag.index(">") + 1]
        for attribute in ("defer", "async", "type="):
            self.assertNotIn(attribute, tag, tag)

    def test_the_javascript_really_left_the_python_source(self):
        html = self._html()
        self.assertNotIn("getElementById('loading').style.display", html)
        self.assertNotIn("box-sizing: border-box", html)

    def test_app_js_parses(self):
        # The payoff: this is the check that was impossible while the JS was
        # a Python string literal.
        if shutil.which("node") is None:
            self.skipTest("node is not installed")
        result = subprocess.run(
            ["node", "--check", str(self.static_dir / "app.js")],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_app_js_kept_single_backslashes_in_regexes(self):
        # The JS body was a non-raw Python string, so the source held "\\s"
        # for what the browser received as "\s". Extracting those 3,356 lines
        # with sed or an editor selection would leave the doubled form, and
        # /[-\\s]/ means "hyphen, backslash or s" -- ISBN normalisation and
        # the date parser would silently misbehave with no error anywhere.
        text = (self.static_dir / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("\\\\", text)
        self.assertIn(r"/[-\s]/g", text)


if __name__ == "__main__":
    unittest.main()
