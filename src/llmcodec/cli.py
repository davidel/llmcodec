"""Command-line interface: ``llmcodec train|encode|decode|roundtrip``."""

from __future__ import annotations

import argparse
from fractions import Fraction

from .candidates import build_selector
from .codec import Codec, NotEncodableError
from .model import TinyLM
from .train import TrainConfig
from .train import train as train_run
from .vocab import DEFAULT_VOCAB


def _make_codec(args) -> Codec:
    model = TinyLM.load(args.model, map_location="cpu")
    selector = build_selector(
        args.mode,
        n=getattr(args, "n", 5),
        max_n=getattr(args, "max_n", 16),
        coverage=getattr(args, "coverage", 0.999),
    )
    return Codec(model, DEFAULT_VOCAB, selector, device="cpu")


def _cmd_train(args) -> int:
    cfg = TrainConfig(
        source=args.source,
        max_stories=args.max_stories,
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        out_dir=args.out_dir,
        device=args.device,
        num_threads=args.num_threads,
        seed=args.seed,
    )
    train_run(cfg)
    return 0


def _cmd_encode(args) -> int:
    codec = _make_codec(args)
    try:
        enc = codec.encode(args.text)
    except NotEncodableError as exc:
        print(f"encode failed: {exc}")
        print("hint: use --mode full (lossless) or a larger --n / --coverage")
        return 1
    print(f"text      : {args.text!r}")
    print(f"value     : {float(enc.value):.12f}")
    print(f"fraction  : {enc.value}")
    print(f"interval  : {enc.interval}")
    print(f"steps     : {enc.n_steps}")
    print(f"selector  : {codec.selector.describe()}")
    return 0


def _cmd_decode(args) -> int:
    codec = _make_codec(args)
    value = Fraction(args.value) if args.exact else Fraction(float(args.value))
    try:
        text = codec.decode(value)
    except (NotEncodableError, RuntimeError) as exc:
        print(f"decode failed: {exc}")
        return 1
    print(f"value     : {float(value):.12f}")
    print(f"text      : {text!r}")
    return 0


def _cmd_roundtrip(args) -> int:
    codec = _make_codec(args)
    try:
        enc = codec.encode(args.text)
        decoded = codec.decode(enc.value)
    except NotEncodableError as exc:
        print(f"roundtrip failed: {exc}")
        print("hint: use --mode full (lossless) or a larger --n / --coverage")
        return 1
    ok = decoded == args.text.lower()
    print(f"text      : {args.text!r}")
    print(f"value     : {float(enc.value):.12f}")
    print(f"decoded   : {decoded!r}")
    print(f"match     : {ok}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmcodec",
        description="Encode sentences as real numbers in [0, 1] with a tiny LM.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train the tiny language model")
    p_train.add_argument("--source", choices=["tinystories", "sample"], default="tinystories")
    p_train.add_argument("--max-stories", type=int, default=200_000)
    p_train.add_argument("--steps", type=int, default=50_000)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--block-size", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--warmup-steps", type=int, default=500)
    p_train.add_argument("--log-every", type=int, default=50)
    p_train.add_argument("--eval-every", type=int, default=500)
    p_train.add_argument("--out-dir", default="runs/default")
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--num-threads", type=int, default=0)
    p_train.add_argument("--seed", type=int, default=1337)
    p_train.set_defaults(func=_cmd_train)

    def add_codec_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", required=True, help="path to a trained model.pt")
        p.add_argument("--mode", choices=["full", "topn", "adaptive"], default="full")
        p.add_argument("--n", type=int, default=5, help="top-N size (mode=topn)")
        p.add_argument("--max-n", type=int, default=16, help="max candidates (mode=adaptive)")
        p.add_argument(
            "--coverage", type=float, default=0.999, help="mass coverage (mode=adaptive)"
        )

    p_enc = sub.add_parser("encode", help="encode a sentence to a real in [0, 1]")
    p_enc.add_argument("text")
    add_codec_args(p_enc)
    p_enc.set_defaults(func=_cmd_encode)

    p_dec = sub.add_parser("decode", help="decode a real in [0, 1] to a sentence")
    p_dec.add_argument("value", help="a real number in [0, 1]")
    p_dec.add_argument("--exact", action="store_true", help="interpret value as an exact fraction")
    add_codec_args(p_dec)
    p_dec.set_defaults(func=_cmd_decode)

    p_rt = sub.add_parser("roundtrip", help="encode then decode and report whether it matches")
    p_rt.add_argument("text")
    add_codec_args(p_rt)
    p_rt.set_defaults(func=_cmd_roundtrip)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
