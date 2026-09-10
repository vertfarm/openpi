"""Run from the OpenPI root: python -m examples.hv1.cli --help."""

import argparse
import json

from .artifacts import read_json
from .workflow import Catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo")
    demo.add_argument("destination")
    export = commands.add_parser("export")
    export.add_argument("--manifest", required=True)
    export.add_argument("--destination", required=True)
    export.add_argument("--allow-synthetic", action="store_true")
    for name in ("scan", "list", "review", "manifest", "serve-review"):
        sub = commands.add_parser(name)
        sub.add_argument("--runtime", required=True)
        sub.add_argument("--raw-root", required=True)
        sub.add_argument("--profile", required=True)
        if name == "review":
            sub.add_argument("--episode", required=True)
            sub.add_argument("--outcome", choices=("success", "failure", "aborted", "unknown"), required=True)
            sub.add_argument("--use", action="store_true")
            sub.add_argument("--reason", default="")
        if name == "manifest":
            sub.add_argument("--output", required=True)
            sub.add_argument("--allow-synthetic", action="store_true")
        if name == "serve-review":
            sub.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import make_demo

        result = make_demo(args.destination)
    elif args.command == "export":
        from .export import export_manifest

        result = export_manifest(args.manifest, args.destination, allow_synthetic=args.allow_synthetic)
    else:
        catalog = Catalog(args.runtime, args.raw_root, read_json(args.profile))
        if args.command == "scan":
            result = catalog.scan()
        elif args.command == "list":
            result = catalog.list()
        elif args.command == "review":
            result = catalog.review(args.episode, args.outcome, args.use, args.reason)
        elif args.command == "manifest":
            result = catalog.manifest(args.output, allow_synthetic=args.allow_synthetic)
        else:
            from .review import make_server

            server = make_server(catalog, args.port)
            print(f"HV1 review: http://127.0.0.1:{server.server_port} (no robot controls)", flush=True)
            try:
                server.serve_forever()
            finally:
                server.server_close()
            return
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
