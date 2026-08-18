# Agent Guidelines for al-folio (v1.x)

**This file is the authoritative entry point for coding agents working in this repo.** Read it before making any change. It is intentionally short and tool-neutral; it links to the one place each longer-form fact lives.

This is a customized personal site built from the `al-folio` v1.x **thin Jekyll starter**. The site owns its content, configuration, and explicitly recorded local overrides. Most runtime — layouts, includes, Sass, Liquid tags, filters, and feature JS — lives in versioned gems published under [`al-org-dev`](https://github.com/al-org-dev).

## Route your change

Find your change on the left; edit only what is on the right.

| Your change                                                                                                              | Goes in                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| Dependency pin, plugin activation, feature flag                                                                          | this repo: `Gemfile` **and** `_config.yml` (both — see below)                                                 |
| Personal content, bibliography, data files                                                                               | this repo: `_pages`, `_bibliography`, `_data`, and site assets                                                |
| Documentation                                                                                                            | this repo: `docs/` (long-form) or this file (agent rules)                                                     |
| Personal-site or cross-plugin integration test                                                                           | this repo: `test/integration_*.sh`                                                                            |
| Plugin catalog metadata                                                                                                  | this repo: `_data/featured_plugins.yml`                                                                       |
| A layout, include, or Sass entrypoint                                                                                    | normally the owning gem; this site has three intentional overrides recorded in `.al-folio-overrides.yml`      |
| A Liquid tag or filter, or what a tag renders                                                                            | the gem that registers it — see the [delegation table](docs/ARCHITECTURE.md#wrapper-to-tag-to-gem-delegation) |
| Feature behavior (search, math, charts, comments, cookies, icons, CV, distill, analytics, images, newsletter, citations) | that feature's gem — see [`docs/BOUNDARIES.md`](docs/BOUNDARIES.md)                                           |
| Component/unit test for gem-owned behavior                                                                               | the owning gem, not here                                                                                      |
| A feature with no existing owner                                                                                         | open a plugin proposal issue first, then a standalone plugin repo                                             |

[`docs/BOUNDARIES.md`](docs/BOUNDARIES.md) is the authoritative area-to-gem table. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) explains how the pieces connect.

## Upstream boundary and local overrides

New reusable runtime behavior belongs in an upstream gem. This site may shadow a gem-owned file only for an intentional site-specific customization, and every such override must be audited and recorded. The current intentional overrides are `_includes/header.liquid`, `_layouts/about.liquid`, and `assets/css/main.scss`.

```
_layouts/   _includes/   _sass/   _scripts/   assets/tailwind/   tailwind.config.js   assets/webfonts/
```

Do not run `npm run lint:style-contract` for this personal site: that upstream-starter check intentionally rejects all local overrides. Do not add a starter-local Tailwind or CSS build pipeline. Instead, review overrides with `bundle exec al-folio upgrade overrides audit` and keep `.al-folio-overrides.yml` current.

These overrides are local to this site and should not be proposed as upstream runtime changes. See [local overrides: your site vs. this repo](docs/ARCHITECTURE.md#local-overrides-your-site-vs-this-repo).

## Three failures that produce no error message

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#failure-modes-that-produce-no-error-message) for the full explanation. The short version:

1. **Features fail silently.** A feature renders only when its gem is loaded _and_ its flag is on _and_ the page opts in. Otherwise the Liquid tag emits an empty string — no warning, no error.
2. **`Gemfile` and `_config.yml` are two lists that must agree.** A plugin in only one of them is inert. Adding or removing a plugin means editing both. Repo dirs use hyphens (`al-folio-core`); gem/plugin ids use underscores (`al_folio_core`).
3. **This site's effective baseurl is empty.** It is published at `https://georgelyu.github.io/`, so use a plain `bundle exec jekyll build`. The local development server is at `http://localhost:4000/`.

## Validated local command set

Run from the repo root, in this order:

```bash
bundle install
npm ci
npm run lint:prettier
bundle exec al-folio upgrade audit --no-fail
bundle exec al-folio upgrade overrides audit --fail-on-stale
bundle exec al-folio upgrade report
bundle exec jekyll build
bash test/integration_site_content.sh
bash test/integration_plugin_toggles.sh
bash test/integration_bootstrap_compat.sh
bash test/integration_upgrade_cli.sh
bash test/integration_css_minify.sh
docker compose up -d
curl -fsS http://127.0.0.1:8080/ >/dev/null
docker compose logs --tail=80
docker compose down
```

`unit-tests.yml` runs the personal-site content check and the integration scripts that do not depend on upstream demo pages. The comments, Distill, new-plugin, style-contract, and visual suites are upstream-starter tests and are not gates for this stripped-down personal site. Docker note: v1 uses `/srv/jekyll/bin/entry_point.sh` and serves from container-local `/tmp/_site` to avoid host bind-mount write deadlocks.

## Before you open a PR

- Keep personal content and site-specific configuration here; route reusable runtime behavior to the owning plugin repo.
- Run `npm run lint:prettier` (Prettier with `@shopify/prettier-plugin-liquid`, `printWidth: 150`). `npx prettier . --write` fixes formatting.
- Keep docs aligned with v1 ownership, and keep each fact in one place — link rather than restate.
- If you create or keep local overrides of plugin-owned files, run `bundle exec al-folio upgrade overrides audit` and commit `.al-folio-overrides.yml` after review.

## Further reading

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the starter and gems fit together, silent failure modes, the v1 config contract, local overrides.
- [`docs/BOUNDARIES.md`](docs/BOUNDARIES.md) — authoritative area-to-gem ownership table and PR triage playbook.
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) — contributor workflow and agent tooling.
- [`docs/README.md`](docs/README.md) — index of all user and maintainer guides.
- `.agents/skills/al-folio-bootstrap/SKILL.md` — new-site setup workflow.
- `.agents/skills/al-folio-v1-migration/SKILL.md` — customized-fork migration and override drift auditing.
- `.codex/skills` and `.claude/skills` are symlinks to `.agents/skills` for agent-specific discovery.
