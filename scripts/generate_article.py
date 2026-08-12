#!/usr/bin/env python3
"""Claude API で Hugo 用のメタ分析記事(Markdown)を自動生成する。

使い方:
    python scripts/generate_article.py --target "DTM オーディオインターフェース 比較"

`ANTHROPIC_API_KEY` は環境変数、またはリポジトリ直下の `.env` から読み込む。
生成物は content/posts/{yyyy-mm-dd}-{slug}.md に保存される。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# --- 定数 ----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
POSTS_DIR = REPO_ROOT / "content" / "posts"

JST = timezone(timedelta(hours=9), "JST")

# claude-3-5-sonnet-20241022 は 2025-10-28 に提供終了(404)。
# 公式の後継が claude-sonnet-5。--model / ANTHROPIC_MODEL で上書き可能。
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_TARGET = "DTM オーディオインターフェース 比較"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_EFFORT = "medium"

SYSTEM_PROMPT = """\
あなたは DTM・オーディオ機材を専門とするテクニカルライター兼アナリストです。
「AUDIO LAB」という、国内外のレビュー・スペック・価格を横断的に収集し、
メタ分析の手法で整理・比較するメディアの記事を執筆します。

# 出力形式(厳守)

Hugo 用の Markdown ファイルの中身だけを出力してください。
説明文、前置き、``` によるコードフェンスでの囲みは一切不要です。
出力は必ず `---` で始まる YAML Front Matter から始めてください。

---
title: "【機材名】メタ分析レビュー：海外・国内評価とスペック徹底比較"
date: {date}
draft: false
slug: "english-kebab-case-slug"
categories: ["DTM機材"]
tags: ["タグ1", "タグ2"]
description: "記事の要約(SEOメタディスクリプション。全角120文字以内)"
---

Front Matter の規則:
- `title`: 【】内は実際の機材名・テーマ名に置き換える。全体で全角40文字以内。
- `date`: 上記の値をそのまま使う。
- `slug`: 記事内容を表す英小文字のケバブケース(例: `audio-interface-comparison-2026`)。
  日本語・記号・空白は使わない。URL に使われるため簡潔にする。
- `categories`: 原則 `["DTM機材"]`。内容に応じて2つ目を足してもよい。
- `tags`: 3〜6個。メーカー名、機材カテゴリ、価格帯、用途などの実用的な語。
- `description`: 検索結果に表示される要約。記事の結論の要点を含める。

# 本文の構成(この順序の H2 見出しで構成する)

## 概要
   何を扱う記事か、なぜ今この機材／テーマが論点なのかを3〜5段落で提示する。

## スペック・機能分析
   主要スペックを Markdown 表で比較する。数値の意味(その差が実使用で何を変えるか)を
   必ず解説する。表を置いて終わりにしない。

## 独自メタ分析：国内外ユーザー評価の傾向
   この記事の核。国内レビューと海外レビューで評価軸がどう食い違うか、
   称賛・不満がどの論点に集中しているかを傾向として整理する。
   割合や件数を挙げる場合は「傾向として」「〜が目立つ」等、
   集計手法の限界がわかる表現にとどめ、精密な統計であるかのように書かない。

## メリット・デメリット
   箇条書き可。デメリットは必ず具体的に書く。「人を選ぶ」等の曖昧な逃げをしない。

## こんな人におすすめ
   想定ユーザー像を2〜4パターン挙げ、それぞれ理由を添える。
   「おすすめしない人」も必ず1パターン含める。

## まとめ
   結論を明示する。判断が分かれる場合は、何を基準に選ぶべきかを示す。

# トーン&マナー

- 客観的・論理的に。断定できることと推測を明確に区別する。
- 音響工学・DTM の専門知識に裏打ちされた考察を行う。専門用語は使ってよいが、
  初出時に一言で補足する(例:「レイテンシ(入力から出力までの遅延)」)。
- 誇張した宣伝文句、絵文字、過剰な感嘆符は使わない。
- 不確かな数値・型番・価格を断定的に書かない。確証がない場合は
  「執筆時点の公表値」「編集部調べ」等の留保を付けるか、言及を避ける。
- 実機を試用した体験談を捏造しない。データとレビューの分析として書く。
- 全体で 3000〜5000 文字程度。
"""

USER_PROMPT_TEMPLATE = """\
以下のテーマで記事を1本執筆してください。

テーマ: {target}

Front Matter の `date` には次の値をそのまま使ってください: {date}
"""


# --- ユーティリティ ------------------------------------------------------


def slugify(text: str) -> str:
    """ASCII 化できる範囲でケバブケースの slug を作る。日本語のみなら空文字。"""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")


def strip_code_fence(text: str) -> str:
    """モデルが全体を ``` で囲んだ場合に剥がす。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def front_matter_value(markdown: str, key: str) -> str | None:
    """先頭の Front Matter から単一のスカラー値を取り出す(YAML パーサ不要の範囲)。"""
    match = re.match(r"^---\r?\n(.*?)\r?\n---", markdown, re.DOTALL)
    if not match:
        return None
    pattern = rf'^{re.escape(key)}:\s*["\']?(.*?)["\']?\s*$'
    found = re.search(pattern, match.group(1), re.MULTILINE)
    return found.group(1).strip() if found else None


def force_date(markdown: str, iso_date: str) -> str:
    """Front Matter の date 行を実際の生成時刻で上書きする。

    モデルは現在日時を知らないため、日付だけはスクリプト側で確定させる。
    """
    match = re.match(r"^(---\r?\n)(.*?)(\r?\n---)", markdown, re.DOTALL)
    if not match:
        return markdown
    head, body, tail = match.groups()
    if re.search(r"^date:", body, re.MULTILINE):
        body = re.sub(r"^date:.*$", f"date: {iso_date}", body, count=1, flags=re.MULTILINE)
    else:
        body = f"date: {iso_date}\n{body}"
    return head + body + tail + markdown[match.end() :]


def build_client() -> anthropic.Anthropic:
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ANTHROPIC_API_KEY が設定されていません。\n"
            "  .env に ANTHROPIC_API_KEY=sk-ant-... を書くか、環境変数に設定してください。\n"
            "  雛形: .env.example"
        )
    return anthropic.Anthropic()


# --- 生成 ----------------------------------------------------------------


def generate(client: anthropic.Anthropic, args: argparse.Namespace, iso_date: str) -> str:
    """Claude を呼び出して Markdown 本文を返す。"""
    # 長文生成のためストリーミングで受ける(非ストリーミングは HTTP タイムアウトの恐れ)。
    with client.messages.stream(
        model=args.model,
        max_tokens=args.max_tokens,
        system=SYSTEM_PROMPT.format(date=iso_date),
        output_config={"effort": args.effort},
        messages=[
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(target=args.target, date=iso_date),
            }
        ],
    ) as stream:
        for _ in stream.text_stream:
            print(".", end="", flush=True)
        message = stream.get_final_message()
    print()

    if message.stop_reason == "refusal":
        detail = getattr(message, "stop_details", None)
        sys.exit(f"モデルが生成を拒否しました (category={getattr(detail, 'category', None)})")
    if message.stop_reason == "max_tokens":
        sys.exit(
            f"max_tokens ({args.max_tokens}) に達して出力が途中で切れました。"
            " --max-tokens を増やすか --effort を下げてください。"
        )

    text = "".join(block.text for block in message.content if block.type == "text")
    if not text.strip():
        sys.exit("モデルからテキストが返りませんでした。")

    usage = message.usage
    print(
        f"  model={message.model} "
        f"in={usage.input_tokens} out={usage.output_tokens} "
        f"cache_read={getattr(usage, 'cache_read_input_tokens', 0)}",
        file=sys.stderr,
    )
    return strip_code_fence(text)


def resolve_slug(markdown: str, args: argparse.Namespace) -> str:
    """--slug > Front Matter の slug > target の ASCII 化 > 既定値、の優先順で決める。"""
    for candidate in (args.slug, front_matter_value(markdown, "slug"), slugify(args.target)):
        if candidate:
            cleaned = slugify(candidate)
            if cleaned:
                return cleaned
    return "article"


# --- エントリポイント ----------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Claude API で Hugo 用のメタ分析記事を生成する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="機材名またはテーマ(例: 'DTM オーディオインターフェース 比較')",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        help="使用する Claude モデル ID",
    )
    parser.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="思考の深さ(高いほど高品質・高コスト)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="出力トークン上限"
    )
    parser.add_argument("--slug", help="ファイル名の slug を明示指定する(省略時は自動)")
    parser.add_argument(
        "--outdir", type=Path, default=POSTS_DIR, help="出力先ディレクトリ"
    )
    parser.add_argument(
        "--force", action="store_true", help="同名ファイルが存在する場合に上書きする"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ファイルに保存せず標準出力に表示する(API 呼び出しは行う)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Windows のコンソールでも日本語を化けさせない
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    now = datetime.now(JST)
    iso_date = now.strftime("%Y-%m-%dT%H:%M:%S+09:00")

    client = build_client()

    print(f"生成中: target={args.target!r} model={args.model} effort={args.effort}")
    markdown = generate(client, args, iso_date)
    markdown = force_date(markdown, iso_date)

    if not markdown.startswith("---"):
        sys.exit("出力が Front Matter (---) で始まっていません。プロンプトを確認してください。")

    if args.dry_run:
        print(markdown)
        return 0

    slug = resolve_slug(markdown, args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    path = args.outdir / f"{now.strftime('%Y-%m-%d')}-{slug}.md"

    if path.exists() and not args.force:
        sys.exit(f"既に存在します: {path}\n上書きするには --force を付けてください。")

    # Hugo とリポジトリの改行コードを揃えるため LF 固定で書き出す
    path.write_text(markdown.rstrip() + "\n", encoding="utf-8", newline="\n")

    title = front_matter_value(markdown, "title") or "(title 取得失敗)"
    print(f"保存しました: {path.relative_to(REPO_ROOT)}")
    print(f"  title: {title}")
    print(f"  {len(markdown)} 文字")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except anthropic.AuthenticationError:
        sys.exit("認証に失敗しました。ANTHROPIC_API_KEY を確認してください。")
    except anthropic.NotFoundError as exc:
        sys.exit(f"モデルが見つかりません: {exc}\n--model で有効なモデル ID を指定してください。")
    except anthropic.RateLimitError:
        sys.exit("レート制限に達しました。しばらく待って再実行してください。")
    except anthropic.APIStatusError as exc:
        sys.exit(f"API エラー ({exc.status_code}): {exc.message}")
    except anthropic.APIConnectionError:
        sys.exit("API に接続できませんでした。ネットワークを確認してください。")
    except KeyboardInterrupt:
        sys.exit(130)
