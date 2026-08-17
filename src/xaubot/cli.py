"""Command-line interface.

Every pipeline stage gets a subcommand so that the whole system is runnable and
inspectable from the shell, without a notebook in the loop. Commands for later
phases (train, backtest, ...) are added as those phases land.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from xaubot import __version__
from xaubot.config.loader import load_config
from xaubot.core.errors import XaubotError
from xaubot.core.logging import setup_logging
from xaubot.core.seeding import seed_everything

app = typer.Typer(
    name="xaubot",
    help="Machine-learning trading system for XAUUSD.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Data ingestion and inspection.", no_args_is_help=True)
app.add_typer(data_app, name="data")
features_app = typer.Typer(help="Feature engineering and point-in-time auditing.", no_args_is_help=True)
app.add_typer(features_app, name="features")

console = Console()

ConfigOpt = Annotated[
    list[Path] | None,
    typer.Option("--config", "-c", help="Extra YAML config file(s), layered in order."),
]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Dotted override, e.g. -s data.source.path=D:/x.csv"),
]
BaseOpt = Annotated[
    Path | None,
    typer.Option("--base", help="Base config file (default: config/base.yaml)."),
]


def _parse_overrides(pairs: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise typer.BadParameter(f"--set expects key=value, got {pair!r}")
        overrides[key.strip()] = value.strip()
    return overrides


def _load(config_files: list[Path] | None, sets: list[str] | None, base: Path | None):  # type: ignore[no-untyped-def]
    return load_config(
        *(config_files or []),
        overrides=_parse_overrides(sets),
        base=base,
    )


@app.callback()
def main(
    log_level: Annotated[str, typer.Option("--log-level", help="DEBUG/INFO/WARNING/ERROR")] = "INFO",
    log_file: Annotated[Path | None, typer.Option("--log-file", help="Also write logs here.")] = None,
) -> None:
    """Global options applied to every subcommand."""
    setup_logging(level=log_level, log_file=log_file, force=True)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"xaubot {__version__}")


@app.command("config-show")
def config_show(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    base: BaseOpt = None,
) -> None:
    """Resolve and print the effective configuration."""
    try:
        cfg = _load(config, set_, base)
    except XaubotError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    from xaubot.config.hashing import config_hash

    console.print_json(cfg.model_dump_json(indent=2))
    console.print(f"\n[dim]config hash: {config_hash(cfg)}[/dim]")


@data_app.command("ingest")
def data_ingest(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    base: BaseOpt = None,
    no_persist: Annotated[bool, typer.Option("--no-persist", help="Analyse only; write nothing.")] = False,
) -> None:
    """Load, validate, clean, resample, and store historical bars."""
    from xaubot.data.pipeline import ingest

    try:
        cfg = _load(config, set_, base)
        seed_everything(cfg.seed)
        result = ingest(cfg, persist=not no_persist)
    except XaubotError as exc:
        console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(1) from exc

    _render_report(result)


def _render_report(result) -> None:  # type: ignore[no-untyped-def]
    report = result.report

    summary = Table(title=f"Ingestion: {report.symbol} {report.timeframe.value}", show_header=False)
    summary.add_column("field", style="cyan")
    summary.add_column("value")
    summary.add_row("source", report.source)
    summary.add_row("sha256", report.source_sha256[:16] + "..." if report.source_sha256 else "-")
    summary.add_row("rows in / out", f"{report.rows_in:,} / {report.rows_out:,}")
    summary.add_row("dropped", f"{report.rows_dropped:,}")
    summary.add_row("range", f"{report.start}  ->  {report.end}")
    if report.gaps is not None:
        summary.add_row("completeness", f"{report.gaps.completeness:.2%}")
        summary.add_row(
            "missing bars",
            f"{report.gaps.missing_bars:,} of {report.gaps.expected_bars:,} expected",
        )
        summary.add_row(
            "largest gap",
            f"{report.gaps.largest_gap_bars} bars @ {report.gaps.largest_gap_start}",
        )
    console.print(summary)

    if report.issues:
        issues = Table(title="Data quality issues")
        issues.add_column("severity")
        issues.add_column("kind", style="cyan")
        issues.add_column("count", justify="right")
        issues.add_column("detail")
        colours = {"INFO": "dim", "WARNING": "yellow", "ERROR": "red"}
        for issue in report.issues:
            style = colours.get(issue.severity.value, "")
            issues.add_row(
                f"[{style}]{issue.severity.value}[/{style}]",
                issue.kind.value,
                f"{issue.count:,}",
                issue.detail,
            )
        console.print(issues)
    else:
        console.print("[green]No data-quality issues found.[/green]")

    if result.context:
        tfs = Table(title="Higher timeframes")
        tfs.add_column("timeframe", style="cyan")
        tfs.add_column("bars", justify="right")
        tfs.add_column("partial", justify="right")
        tfs.add_column("range")
        for tf, frame in result.context.items():
            partial = int(frame.df["is_partial"].sum()) if "is_partial" in frame.df else 0
            tfs.add_row(tf.value, f"{len(frame):,}", f"{partial:,}", f"{frame.start} -> {frame.end}")
        console.print(tfs)

    if result.written:
        console.print(f"[green]Wrote {len(result.written)} partition files.[/green]")

    if report.gaps and report.gaps.gaps_over_threshold:
        gaps = Table(title="Largest gaps (feed outages, calendar-adjusted)")
        gaps.add_column("start")
        gaps.add_column("end")
        gaps.add_column("bars", justify="right")
        for start, end, bars in report.gaps.gaps_over_threshold[:10]:
            gaps.add_row(start, end, str(bars))
        console.print(gaps)


@data_app.command("summary")
def data_summary(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    base: BaseOpt = None,
) -> None:
    """Show what is currently in the bar store."""
    from xaubot.data.store import store_summary

    cfg = _load(config, set_, base)
    frames = []
    for root, label in ((cfg.paths.canonical(), "canonical"), (cfg.paths.resampled(), "resampled")):
        summary = store_summary(root)
        if not summary.empty:
            summary.insert(0, "store", label)
            frames.append(summary)

    if not frames:
        console.print("[yellow]Store is empty. Run `xaubot data ingest` first.[/yellow]")
        return

    import pandas as pd

    combined = pd.concat(frames, ignore_index=True)
    table = Table(title="Bar store")
    for column in combined.columns:
        table.add_column(str(column))
    for _, row in combined.iterrows():
        table.add_row(*[str(v) for v in row.tolist()])
    console.print(table)


def _load_bars(cfg):  # type: ignore[no-untyped-def]
    """Read base and context bars from the store, with a helpful error."""
    from xaubot.core.errors import StorageError
    from xaubot.data.store import read_bars

    try:
        base = read_bars(cfg.paths.canonical(), cfg.data.symbol, cfg.data.base_timeframe)
        context = {
            tf: read_bars(cfg.paths.resampled(), cfg.data.symbol, tf) for tf in cfg.data.context_timeframes
        }
    except StorageError as exc:
        console.print(f"[red]{exc}[/red]\n[yellow]Run `xaubot data ingest` first.[/yellow]")
        raise typer.Exit(1) from exc
    return base, context


@features_app.command("build")
def features_build(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    base: BaseOpt = None,
    no_persist: Annotated[bool, typer.Option("--no-persist", help="Compute only; write nothing.")] = False,
    audit: Annotated[
        bool, typer.Option("--audit/--no-audit", help="Run the replay-equivalence audit.")
    ] = True,
) -> None:
    """Compute the feature matrix from stored bars."""
    from xaubot.data.pipeline import build_calendar
    from xaubot.features.engine import build_engine
    from xaubot.features.pit_audit import audit_replay_equivalence
    from xaubot.features.store import write_features

    cfg = _load(config, set_, base)
    seed_everything(cfg.seed)
    bars, context = _load_bars(cfg)

    engine = build_engine(cfg, build_calendar(cfg))
    required = engine.required_warmup_bars(cfg.data.base_timeframe)

    try:
        matrix = engine.transform(bars, context)
    except XaubotError as exc:
        console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(1) from exc

    summary = Table(title="Feature matrix", show_header=False)
    summary.add_column("field", style="cyan")
    summary.add_column("value")
    summary.add_row("fs_version", matrix.manifest.fs_version)
    summary.add_row("features", f"{matrix.manifest.n_features:,}")
    summary.add_row("bars", f"{len(matrix):,}")
    summary.add_row("range", f"{matrix.start}  ->  {matrix.end}")
    summary.add_row("warmup required", f"{required:,} base bars")
    summary.add_row("warmup dropped", f"{matrix.warmup_dropped:,} bars")
    console.print(summary)

    groups = Table(title="Features by group")
    groups.add_column("group", style="cyan")
    groups.add_column("count", justify="right")
    for name, names in sorted(matrix.manifest.by_group().items()):
        groups.add_row(name, str(len(names)))
    console.print(groups)

    nan = matrix.nan_report()
    problem = nan[nan > 0.02]
    if len(problem):
        table = Table(title=f"Columns with >2% NaN ({len(problem)} of {len(nan)})")
        table.add_column("feature", style="cyan")
        table.add_column("nan %", justify="right")
        for name, value in problem.head(15).items():
            table.add_row(str(name), f"{value:.1%}")
        console.print(table)
        console.print(
            "[dim]NaN here is usually insufficient history for a long lookback, not a bug. "
            "The dataset stage drops such columns using TRAIN-fold statistics only.[/dim]"
        )

    if audit:
        console.print("\n[cyan]Running replay-equivalence audit...[/cyan]")
        try:
            result = audit_replay_equivalence(engine, bars, context, n_cutoffs=4, raise_on_failure=True)
        except XaubotError as exc:
            console.print(f"[red]LEAKAGE DETECTED\n{exc}[/red]")
            raise typer.Exit(2) from exc
        console.print(f"[green]{result.summary()}[/green]")

    if not no_persist:
        path = write_features(matrix, cfg.paths.data_root / cfg.paths.features_dir, cfg.data.symbol)
        console.print(f"[green]Wrote feature set to {path}[/green]")


@features_app.command("audit")
def features_audit(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    base: BaseOpt = None,
    cutoffs: Annotated[int, typer.Option("--cutoffs", help="Truncation points to test.")] = 6,
) -> None:
    """Run the point-in-time audits without materialising a feature set."""
    from pathlib import Path as _Path

    from xaubot.data.pipeline import build_calendar
    from xaubot.features.engine import build_engine
    from xaubot.features.pit_audit import audit_replay_equivalence, scan_source_for_lookahead

    cfg = _load(config, set_, base)
    package = _Path(__file__).parent / "features"

    findings = scan_source_for_lookahead(package)
    if findings:
        table = Table(title="Static look-ahead scan")
        table.add_column("location", style="cyan")
        table.add_column("construct")
        table.add_column("why it leaks")
        for finding in findings:
            table.add_row(f"{finding.path}:{finding.line}", finding.construct, finding.reason)
        console.print(table)
    else:
        console.print("[green]Static scan clean: no forward-looking constructs in features/.[/green]")

    bars, context = _load_bars(cfg)
    engine = build_engine(cfg, build_calendar(cfg))
    try:
        result = audit_replay_equivalence(engine, bars, context, n_cutoffs=cutoffs, raise_on_failure=False)
    except XaubotError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if result.passed:
        console.print(f"[green]{result.summary()}[/green]")
        return

    table = Table(title="Features that changed under truncated replay")
    table.add_column("feature", style="red")
    table.add_column("failed cutoffs", justify="right")
    for name, count in sorted(result.mismatches.items(), key=lambda item: -item[1])[:25]:
        table.add_row(name, f"{count}/{result.cutoffs_tested}")
    console.print(table)
    console.print(
        "[red]These features depend on data that would not exist at decision time. "
        "Every downstream result is invalid until this is fixed.[/red]"
    )
    raise typer.Exit(2)


@features_app.command("list")
def features_list(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    base: BaseOpt = None,
) -> None:
    """List materialised feature sets in the store."""
    from xaubot.features.store import list_feature_sets

    cfg = _load(config, set_, base)
    frame = list_feature_sets(cfg.paths.data_root / cfg.paths.features_dir)
    if frame.empty:
        console.print("[yellow]No feature sets built. Run `xaubot features build`.[/yellow]")
        return

    table = Table(title="Feature sets")
    for column in frame.columns:
        table.add_column(str(column))
    for _, row in frame.iterrows():
        table.add_row(*[str(value) for value in row.tolist()])
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
