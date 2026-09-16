"""Make README links portable in the disposable checkout used to build PyPI wheels."""

import argparse
from pathlib import Path
import re
from urllib.parse import quote, urlsplit


def prepare_readme(readme, repository_url, revision):
    """Resolve local Markdown links against the exact source revision being packaged.

    GitHub resolves relative README links itself; PyPI has no repository context.
    Leave external URLs, heading anchors, and the existing remote images intact.
    Only the wheel-build checkout is rewritten, keeping one maintained README.
    """
    base = repository_url.rstrip("/")
    if urlsplit(base).scheme != "https":
        raise ValueError("The repository URL must use HTTPS")
    revision = quote(revision, safe="")

    def replace_link(match):
        target = match.group(2)
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match.group(0)
        path = readme.parent / parsed.path
        if not path.exists():
            raise ValueError(f"README link target does not exist: {target}")
        kind = "tree" if path.is_dir() else "blob"
        url = f"{base}/{kind}/{revision}/{quote(parsed.path, safe='/')}"
        if parsed.query:
            url += "?" + parsed.query
        if parsed.fragment:
            url += "#" + parsed.fragment
        return f"[{match.group(1)}]({url})"

    source = readme.read_text(encoding="utf-8")
    # The maintained README uses inline Markdown links without titles or spaces
    # in their targets; HTML images already use absolute remote URLs.
    result = re.sub(r"\[([^\]\n]+)\]\(([^\s)]+)\)", replace_link, source)
    readme.write_text(result, encoding="utf-8")


def main():
    """Prepare the supplied build-checkout README without fetching remote content."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("readme", type=Path)
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    prepare_readme(args.readme, args.repository_url, args.revision)


if __name__ == "__main__":
    main()
