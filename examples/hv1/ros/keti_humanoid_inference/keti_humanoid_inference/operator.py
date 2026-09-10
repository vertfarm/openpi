"""Local UNIX socket only. No network ARM service."""

import argparse
import socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("command", choices=("ARM", "STOP"))
    parser.add_argument("--review-id", default="")
    args = parser.parse_args()
    if args.command == "ARM" and not args.review_id:
        parser.error("ARM needs the reviewed field profile id")
    message = args.command + (" " + args.review_id if args.command == "ARM" else "")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(3)
        client.connect(args.socket)
        client.sendall((message + "\n").encode())
        print(client.recv(4096).decode().strip())


if __name__ == "__main__":
    main()
