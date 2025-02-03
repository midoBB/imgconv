#!/usr/bin/env python3
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Generator, Tuple

import click
from magic import Magic
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

ERROR_LOG = "imgopt_errors.log"
OPTIM_MARKER = "optim_"
current_process = None
console = Console()


def signal_handler(sig, frame):
    """Handle signals and clean up resources"""
    global current_process
    if current_process:
        current_process.terminate()
    console.print("\nExiting due to interrupt...")
    sys.exit(1)


def format_size(bytes: float) -> str:
    """Convert bytes to a human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(bytes) < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"


class IsOptimized(Enum):
    NOT_IMAGE = 0
    NOT_OPTIMIZED = 1
    IS_OPTIMIZED = 2
    ERROR = 3


def is_optimized(file_path: Path) -> IsOptimized:
    """Check if file is already optimized"""
    if file_path.name.startswith(OPTIM_MARKER):
        return IsOptimized.IS_OPTIMIZED

    mime = Magic(mime=True)
    mimetype = mime.from_file(file_path)
    if not mimetype.startswith("image/"):
        return IsOptimized.NOT_IMAGE

    try:
        comment = (
            subprocess.check_output(
                ["identify", "-format", "%[comment]", str(file_path)],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return (
            IsOptimized.IS_OPTIMIZED
            if "IMGOPT_PROTECTED" in comment
            else IsOptimized.NOT_OPTIMIZED
        )
    except subprocess.CalledProcessError:
        return IsOptimized.ERROR


def image_generator(input_path: Path, process_all: bool) -> Generator[Path, None, None]:
    """Generate all files to process, including skipped ones"""
    if process_all:
        for f in os.listdir(input_path):
            if os.path.isfile(os.path.join(input_path, f)):
                yield Path(input_path) / f
    else:
        for path in input_path:
            yield path


def process_image(
    file_path: Path,
    jpeg_quality: int,
    max_dimension: int,
    resize_method: str,
) -> Tuple[bool, int]:
    """Process a single image file"""
    global current_process
    temp_file = None
    try:
        mime = Magic(mime=True)
        mimetype = mime.from_file(file_path)

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            dir=file_path.parent, delete=False, suffix=file_path.suffix
        )
        temp_path = Path(temp_file.name)
        temp_file.close()

        # Resize and optimize
        if mimetype == "image/jpeg":
            cmd = [
                "magick",
                str(file_path),
                "-resize",
                f"{max_dimension}x{max_dimension}^>",
                "-filter",
                resize_method,
                "-strip",
                "-interlace",
                "Plane",
                "-quality",
                str(jpeg_quality),
                "-set",
                "comment",
                "IMGOPT_PROTECTED",
                str(temp_path),
            ]
        elif mimetype == "image/png":
            cmd = [
                "magick",
                str(file_path),
                "-resize",
                f"{max_dimension}x{max_dimension}^>",
                "-filter",
                resize_method,
                str(temp_path),
            ]
            current_process = subprocess.run(cmd, check=True)
            subprocess.run(
                ["oxipng", "-o6", "--strip", "all", str(file_path)], check=True
            )
            cmd = [
                "magick",
                str(temp_path),
                "-set",
                "comment",
                "IMGOPT_PROTECTED",
                str(temp_path),
            ]
        elif mimetype == "image/gif":
            cmd = [
                "magick",
                str(file_path),
                "-resize",
                f"{max_dimension}x{max_dimension}^>",
                str(temp_path),
            ]
            current_process = subprocess.run(cmd, check=True)
            subprocess.run(
                [
                    "gifsicle",
                    "-O3",
                    "--lossy=50",
                    str(temp_path),
                    "-o",
                    str(temp_path),
                ],
                check=True,
            )
        elif mimetype == "image/webp":
            cmd = [
                "magick",
                str(file_path),
                "-resize",
                f"{max_dimension}x{max_dimension}^>",
                "-filter",
                resize_method,
                "-strip",
                "-quality",
                str(jpeg_quality),
                "-set",
                "comment",
                "IMGOPT_PROTECTED",
                str(temp_path),
            ]
        else:
            raise ValueError(f"Unsupported MIME type: {mimetype}")

        current_process = subprocess.run(cmd, check=True)

        # Compare file sizes
        original_size = file_path.stat().st_size
        new_size = temp_path.stat().st_size

        if new_size < original_size:
            optimized_path = file_path.with_name(f"{OPTIM_MARKER}{file_path.name}")
            temp_path.rename(optimized_path)
            file_path.unlink()
            optimized_path.rename(file_path)
            return True, original_size - new_size
        else:
            temp_path.unlink()
            return False, 0

    except Exception as e:
        if temp_path and temp_path.exists():
            temp_path.unlink()
        raise e
    finally:
        current_process = None


def log_error(file_path: Path, error: str):
    """Log errors to the error log file"""
    with open(ERROR_LOG, "a") as f:
        f.write(f"Error processing {file_path}:\n{error}\n")
        f.write("-" * 40 + "\n")


def check_required_tools():
    """Check if required tools are installed and available in the system PATH."""
    required_tools = {
        "magick": "ImageMagick (required for image processing)",
        "oxipng": "oxipng (required for PNG optimization)",
        "gifsicle": "gifsicle (required for GIF optimization)",
    }
    missing_tools = []

    for tool, description in required_tools.items():
        if not shutil.which(tool):
            missing_tools.append(f"{tool} ({description})")

    if missing_tools:
        console.print(
            "[red]Error: The following required tools are not installed:[/red]"
        )
        for tool in missing_tools:
            console.print(f"  - {tool}")
        console.print("\nPlease install the missing tools and try again.")
        sys.exit(1)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("input_path", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("-j", "--jpeg-quality", type=click.IntRange(1, 100), default=85)
@click.option("-d", "--max-dimension", type=int, default=1200)
@click.option(
    "-a", "--all", "process_all", is_flag=True, help="Process all images in directory"
)
def main(input_path, jpeg_quality, max_dimension, process_all):
    """Image optimization tool with similar functionality to the bash script"""
    check_required_tools()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if Path(ERROR_LOG).exists():
        Path(ERROR_LOG).unlink()

    # Validate input
    if not input_path and not process_all:
        raise click.UsageError("Must specify input files or use --all")
    if process_all and input_path:
        raise click.UsageError("Cannot specify both --all and input files")

    base_path = Path.cwd() if process_all else Path(input_path[0]).parent

    # Count total files first
    def count_total_files():
        if process_all:
            return len(
                [
                    f
                    for f in os.listdir(base_path)
                    if os.path.isfile(os.path.join(base_path, f))
                ]
            )
        return len(input_path)

    total_files = count_total_files()
    if total_files == 0:
        console.print("No files to process")
        return

    # Prepare progress display
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[white]{task.completed}/{task.total} files"),
        TimeElapsedColumn(),
    )

    processed = 0
    errors = 0
    total_saved = 0

    # Use Live to display both the progress and the conversion queue
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel

    def get_queue_display(current_index, files, statuses):
        """Generate queue display lines"""
        start = max(0, current_index - 2)
        end = min(len(files), current_index + 3)
        lines = []

        for idx in range(start, end):
            if idx >= len(files):
                continue
            if idx < current_index:
                status = statuses.get(idx, "pending")
                if status == "success":
                    prefix = "[green]✓[/green]"
                elif status == "error":
                    prefix = "[red]✗[/red]"
                elif status == "skipped":
                    prefix = "[cyan]S[/cyan]"
                else:
                    prefix = "[white]?[/white]"
            elif idx == current_index:
                prefix = "[yellow]→[/yellow]"
            else:
                prefix = "  "
            lines.append(f"{prefix} {files[idx].name}")
        return lines

    with Live(
        Group(
            progress,
            Panel(
                "Queue initializing...",
                title="Optimization Queue",
                border_style="magenta",
            ),
        ),
        refresh_per_second=4,
        console=console,
    ) as live:
        total_task = progress.add_task("[cyan]Optimizing images", total=total_files)
        statuses = {}

        # Generator for iterating through all files
        files_generator = list(
            image_generator(base_path if process_all else input_path, process_all)
        )

        for i, file_path in enumerate(files_generator):
            # Update queue display
            queue_display = get_queue_display(i, files_generator, statuses)
            queue_display.append("")
            queue_display.append(f"Total space saved: {format_size(total_saved)}")
            queue_lines = "\n".join(queue_display)
            queue_panel = Panel(
                queue_lines, title="Optimization Queue", border_style="magenta"
            )
            live.update(Group(progress, queue_panel))

            if is_optimized(file_path) == IsOptimized.NOT_OPTIMIZED:
                try:
                    success, saved = process_image(
                        file_path,
                        jpeg_quality,
                        max_dimension,
                        "Lanczos",
                    )
                    if success:
                        total_saved += saved
                        processed += 1
                        statuses[i] = "success"
                    else:
                        statuses[i] = "skipped"
                except Exception as e:
                    log_error(file_path, str(e))
                    errors += 1
                    statuses[i] = "error"
            else:
                statuses[i] = "skipped"

            progress.update(total_task, advance=1)

    # Print summary
    console.print(f"\nProcessed {processed}/{total_files} files successfully")
    console.print(f"Total space saved: {format_size(total_saved)}")
    if errors > 0:
        console.print(f"Encountered {errors} errors - see {ERROR_LOG} for details")
    else:
        if Path(ERROR_LOG).exists():
            Path(ERROR_LOG).unlink()


if __name__ == "__main__":
    main()
