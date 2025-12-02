from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple, Union

import yaml
from PIL import Image


def list_files(
    folder: str | Path, suffixes: Optional[List[str]] = None, recursive: bool = False
) -> List[Path]:
    """
    List all files in a folder, optionally filtering by multiple file extensions.

    Args:
        folder (str | Path): Path to the folder.
        suffixes (List[str], optional): List of extensions to filter
            by (e.g., ['.png', '.jpg']).
        recursive (bool, optional): Whether to search subfolders recursively.
            Defaults to False.

    Returns:
        List[Path]: List of file paths matching the criteria.
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    # Choose the iterator (recursive or not)
    files_iter = folder.rglob('*') if recursive else folder.glob('*')

    # Filter by suffixes
    if suffixes:
        suffixes = [
            s.lower() if s.startswith('.') else f'.{s.lower()}' for s in suffixes
        ]
        files = [
            f for f in files_iter if f.is_file() and f.suffix.lower() in suffixes
        ]
    else:
        files = [f for f in files_iter if f.is_file()]

    return sorted(files)


def read_image_rgb(path: Union[str, Path]) -> Image.Image:
    """
    Reads an image file and converts it to RGB format using Pillow.

    Args:
        path (str | Path): Path to the image file.

    Returns:
        PIL.Image.Image: The image converted to RGB format.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the image cannot be opened or read.
    """
    path = Path(path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except OSError as e:
        raise OSError(f"Failed to read image {path}: {e}") from e


def save_image_rgb(
        image: Image.Image, 
        path: Union[str, Path], 
        format: str | None = None
    ) -> None:
    """
    Saves a PIL RGB image to a file.

    Args:
        image (PIL.Image.Image): The image to save. Should be in 'RGB' mode.
        path (str | Path): The output file path.
        format (str | None): Optional image format (e.g., 'PNG', 'JPEG').
                             If None, Pillow infers from file extension.

    Raises:
        ValueError: If the provided image is not a PIL.Image.Image or not in RGB mode.
        OSError: If saving fails.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL.Image.Image instance.")
    if image.mode != "RGB":
        raise ValueError(f"Image must be in RGB mode (got '{image.mode}')")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        image.save(path, format=format)
    except OSError as e:
        raise OSError(f"Failed to save image to {path}: {e}") from e


def save_images(
    images: Union[List[Image.Image], Dict[str, Image.Image]],
    out_dir: Union[str, Path],
    prefix: str = "image",
    ext: str = "png",
    overwrite: bool = True,
) -> List[Path]:
    """
    Save a list or dictionary of PIL RGB images to a directory.

    Args:
        images (list[Image.Image] | dict[str, Image.Image]):
            List or dict of PIL.Image.Image objects to save.
        out_dir (str | Path):
            Target directory (created if missing).
        prefix (str):
            Filename prefix (used only for list input).
        ext (str):
            File extension, e.g. 'png', 'jpg'.
        overwrite (bool):
            Whether to overwrite existing files (default: True).

    Returns:
        list[Path]: List of paths where images were saved.

    Raises:
        ValueError: If inputs are invalid or images not in RGB mode.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    ext = ext.lstrip(".").lower()

    pairs: Iterable[Tuple[str, Image.Image]]
    if isinstance(images, dict):
        pairs = images.items()
    elif isinstance(images, list):
        pairs = [(f"{prefix}_{i}", img) for i, img in enumerate(images)]
    else:
        raise ValueError(f"Expected list or dict of PIL.Image, got {type(images)}")

    for name, img in pairs:
        if not isinstance(img, Image.Image):
            raise ValueError(f"Item '{name}' is not a PIL.Image.Image instance.")
        if img.mode != "RGB":
            img = img.convert("RGB")

        path = out_dir / f"{name}.{ext}"
        if not overwrite and path.exists():
            i = 1
            while (out_dir / f"{name}_{i}.{ext}").exists():
                i += 1
            path = out_dir / f"{name}_{i}.{ext}"

        save_image_rgb(img, path)
        saved_paths.append(path)

    return saved_paths


def dict_to_namespace(d: dict):
    """Convert a nested dictionary to a SimpleNamespace.

    Recursively transforms dictionaries and lists into objects that support
    attribute-style access.

    Args:
        d (dict): Input dictionary.

    Returns:
        SimpleNamespace | list | Any: Converted namespace, list, or value.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


def read_yaml_config(
        path: Union[str, Path], args: Optional[object] = None
    ) -> SimpleNamespace:
    """Load a YAML config file and return it as a SimpleNamespace.

    Recursively converts the YAML dictionary into an object that supports
    attribute-style access. Optionally attaches command-line arguments under
    the `args` attribute.

    Args:
        path (str | Path): Path to the YAML configuration file.
        args (object, optional): CLI arguments (e.g., argparse.Namespace or dict)
            to attach to the returned config.

    Returns:
        SimpleNamespace: Parsed configuration with attribute access.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        config_dict = yaml.safe_load(f) or {}

    # Convert config dict to SimpleNamespace
    config = dict_to_namespace(config_dict)

    # Optionally add CLI args
    if args is not None:
        if isinstance(args, dict):
            args_dict = args
        else:
            # handles argparse.Namespace or any object with __dict__
            args_dict = vars(args)
        config.args = SimpleNamespace(**args_dict)

    return config
