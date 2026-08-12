#!/usr/bin/env python3
"""Claude API で Hugo 用の DTM 機材レビュー記事(Markdown)を自動生成する。

使い方:
    python scripts/generate_article.py --target "DTM向けオーディオインターフェースおすすめ5選 2026"

`ANTHROPIC_API_KEY` は環境変数、またはリポジトリ直下の `.env` から読み込む。
生成物は content/posts/{yyyy-mm-dd}-{slug}.md に保存される。

画像 / YouTube の扱い:
    モデルは実在する画像 URL や動画 ID を知らないため、必ずプレースホルダを出力させる。
    既定 (--placeholders comment) では、それらを HTML コメントの TODO に変換してから
    保存する。壊れた埋め込みが本番に出るのを防ぐため。
    --placeholders raw を付けるとモデルの出力をそのまま保存する。
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

# 製品カテゴリの判断精度を優先して Sonnet 5 を既定にする。
# コストを抑えたいときは --model claude-haiku-4-5-20251001 を付ける
# (約1/3のコストだが、機種選定の妥当性は落ちる)。
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_TARGET = "DTM向けオーディオインターフェースおすすめ5選 2026"
# 本文1,500〜2,000文字 + 表 + 思考分の余裕。上限値は実際の生成分しか課金されない
# ため、途中切れ(max_tokens 到達)による無駄打ちを避けて余裕を取る。
DEFAULT_MAX_TOKENS = 6000
DEFAULT_EFFORT = "low"
DEFAULT_PRODUCTS = 5

# 本文の目標文字数(Front Matter とコメントを除く)
TARGET_CHARS = (1500, 2000)
CHARS_HARD_MAX = 2600

# effort パラメータは Claude 5 系のみ対応。Haiku 4.5 に送ると 400 になる。
EFFORT_CAPABLE = re.compile(r"(?:opus|sonnet|fable)-5")

VIDEO_PLACEHOLDER = "VIDEO_ID"
IMAGE_DIR = "/images/products"

SYSTEM_PROMPT = """\
あなたは DTM 機材メディア「AUDIO LAB」の編集者です。
ニュース速報の文体で、結論から先に短く書きます。

# 絶対条件

- 本文は全体で{lo}〜{hi}文字。超えたら説明を削る。長文の地の文は禁止。
- 1段落は最大2文。説明が長くなるなら箇条書きに置き換える。
- 禁止表現: 挨拶(こんにちは)、前置き(〜について解説します/見ていきましょう)、
  締め(いかがでしたでしょうか/結論として/まとめると)、記事自己言及(本記事では/この記事では)、
  曖昧な推量止め(〜ではないでしょうか/〜と言えるでしょう)、絵文字(🎯を除く)、煽り表現。
- 敬体(です・ます)で統一。
- 型番・価格・スペックに確証がなければ「公表なし」と書く。数値を推測で埋めない。
  存在しない製品を作らない。実機を試用した体験談を書かない。

# 出力形式

Markdown 本文のみを出力する。コードフェンスで囲まない。Front Matter から始める。

---
title: "記事タイトル(全角40文字以内)"
date: {date}
draft: false
slug: "english-kebab-case"
categories: ["DTM機材"]
tags: ["タグ1", "タグ2"]
description: "SEO用要約(全角120文字以内)"
---

# 冒頭(見出しなし、この順)

1. アイキャッチ画像: `![記事テーマ](/images/products/eyecatch.jpg)`
2. 概要を2文だけ。
3. ポイント3選の引用枠:

> **この記事のポイント**
>
> - ポイント1
> - ポイント2
> - ポイント3

4. PR表記を1行: 本記事にはアフィリエイトリンクが含まれます。

# 製品セクション(製品数 = {products})

**製品を1つずつ独立した H2 セクションで扱う。まとめて論じるのは禁止。**
{products}製品なら H2 を{products}個作る。見出しは `## 1. メーカー名 製品名` の形式で通し番号を振る。

各セクションは以下の順で構成する。

画像と動画(実在する URL や動画 ID を創作せず、下記をそのまま使う):

![メーカー名 製品名](/images/products/PRODUCT_SLUG.jpg)

{{{{< youtube VIDEO_ID >}}}}

`PRODUCT_SLUG` は製品名の英小文字ケバブケース(例: `focusrite-scarlett-2i2`)に置き換える。
`VIDEO_ID` は置き換えず、この文字列のまま出力する。

次に `### 特徴`。2行程度の概要のあと、太字箇条書き3〜4項目:

- **プリアンプ**: 短い説明
- **接続**: 短い説明

次に `### スペック・動作環境`。Markdown 表で記載する。
行は「入出力 / 解像度 / 対応OS / 接続方式 / 電源 / 実勢価格」。

次に引用枠を1〜2行(見出しなし):

> **🎯 こんな人におすすめ**
>
> - 用途を1行で

セクション末尾にアフィリエイト行を置く。**URL 内の空白は必ず `+` に置き換える。**

[Amazonで見る](https://www.amazon.co.jp/s?k=製品名) | [サウンドハウスで見る](https://www.soundhouse.co.jp/search/index?s_ak=製品名)

# 末尾

`## 選び方` の H2 を置き、用途別に1行ずつの箇条書きだけを書く。地の文は不要。
"""

USER_PROMPT_TEMPLATE = """\
以下のテーマで記事を1本執筆してください。

テーマ: {target}
取り上げる製品数: {products}
本文の文字数: {lo}〜{hi}文字(厳守)

Front Matter の `date` には次の値をそのまま使ってください: {date}
"""


def render_system_prompt(products: int, iso_date: str) -> str:
    return SYSTEM_PROMPT.format(
        date=iso_date, products=products, lo=TARGET_CHARS[0], hi=TARGET_CHARS[1]
    )


def render_user_prompt(target: str, products: int, iso_date: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        target=target,
        products=products,
        date=iso_date,
        lo=TARGET_CHARS[0],
        hi=TARGET_CHARS[1],
    )


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
    lines = stripped.splitlines()[1:]
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


# --- 後処理 --------------------------------------------------------------


def fix_affiliate_urls(markdown: str) -> str:
    """Amazon / サウンドハウスの検索 URL 内の空白を + に直す。

    空白が残っていると Markdown のリンク記法が壊れるため。
    """

    def repl(match: re.Match[str]) -> str:
        return match.group(0).replace(" ", "+").replace("　", "+")

    return re.sub(
        r"https://www\.(?:amazon\.co\.jp|soundhouse\.co\.jp)/[^)\n]*", repl, markdown
    )


def neutralize_media(markdown: str) -> tuple[str, int, int]:
    """未確定の画像・YouTube 埋め込みを HTML コメントの TODO に退避する。

    モデルは実在する画像 URL も動画 ID も知らない。そのまま出すと本番サイトに
    404 画像と壊れた iframe が並ぶため、編集部が差し替えるまでコメント化する。
    戻り値は (変換後 Markdown, 画像件数, 動画件数)。
    """
    videos = 0
    images = 0

    def repl_video(match: re.Match[str]) -> str:
        nonlocal videos
        videos += 1
        # {{</* ... */>}} は Hugo がショートコードとして実行せず、
        # リテラル文字列として扱う。HTML コメント内なので画面には出ない。
        return (
            "<!-- TODO(動画): 実在する YouTube 動画 ID を確認し、"
            "下記のコメントを解除して使用\n"
            f"{{{{</* youtube {VIDEO_PLACEHOLDER} */>}}}}\n"
            "-->"
        )

    def repl_image(match: re.Match[str]) -> str:
        nonlocal images
        images += 1
        alt, path = match.group(1), match.group(2)
        return (
            f"<!-- TODO(画像): static{path} を用意し、下記のコメントを解除して使用\n"
            f"![{alt}]({path})\n"
            "-->"
        )

    markdown = re.sub(r"\{\{<\s*youtube\s+[^>\n]*?\s*>\}\}", repl_video, markdown)
    markdown = re.sub(
        rf"!\[([^\]]*)\]\(({re.escape(IMAGE_DIR)}/[^)\s]+)\)", repl_image, markdown
    )
    return markdown, images, videos


def lint(markdown: str, products: int) -> list[str]:
    """文体・構成のチェック。ビルドは止めず警告として返す。"""
    warnings: list[str] = []

    body = re.sub(r"^---\r?\n.*?\r?\n---", "", markdown, flags=re.DOTALL)

    banned = [
        "こんにちは",
        "解説していきます",
        "解説します",
        "見ていきましょう",
        "紹介していきます",
        "いかがでしたでしょうか",
        "いかがでしょうか",
        "結論として",
        "まとめると",
        "本記事では",
        "この記事では",
        "ではないでしょうか",
        "と言えるでしょう",
    ]
    for phrase in banned:
        if phrase in body:
            warnings.append(f"禁止表現が残っています: 「{phrase}」")

    if "本記事にはアフィリエイトリンクが含まれます" not in body:
        warnings.append("PR表記がありません")

    # 本文の実文字数(コメント・画像・リンクURL・表を除いた地の文+箇条書き)
    prose = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    prose = re.sub(r"\[[^\]]*\]\([^)]*\)", "", prose)
    prose = re.sub(r"^\|.*$", "", prose, flags=re.MULTILINE)
    chars = len(re.sub(r"\s", "", prose))
    lo, hi = TARGET_CHARS
    if chars > CHARS_HARD_MAX:
        warnings.append(f"本文が {chars} 文字。目標 {lo}〜{hi} を大きく超過")
    elif chars < lo * 0.7:
        warnings.append(f"本文が {chars} 文字。目標 {lo}〜{hi} に対して不足")

    # アイキャッチ = 最初の H2 より前に置かれた画像。ファイル名は問わない。
    lead = body.split("\n## ", 1)[0]
    if IMAGE_DIR not in lead:
        warnings.append("冒頭のアイキャッチ画像がありません")

    if "**この記事のポイント**" not in body:
        warnings.append("冒頭のポイント3選の引用枠がありません")

    h2_count = len(re.findall(r"^## ", body, re.MULTILINE))
    if h2_count < products:
        warnings.append(f"H2 見出しが {h2_count} 個。製品数 {products} に対して少ない可能性")

    amazon = len(re.findall(r"amazon\.co\.jp/s\?k=", body))
    if amazon < products:
        warnings.append(f"Amazon 検索リンクが {amazon} 個。製品数 {products} に不足")

    quote = len(re.findall(r"^> \*\*🎯", body, re.MULTILINE))
    if quote < products:
        warnings.append(f"🎯 引用ボックスが {quote} 個。製品数 {products} に不足")

    # 3文以上の段落を検出(表・箇条書き・引用・コメントは除外)
    long_paragraphs = 0
    for block in re.split(r"\n\s*\n", body):
        block = block.strip()
        if not block or block.startswith(("#", "|", "-", ">", "<!--", "!", "[")):
            continue
        if len(re.findall(r"。", block)) >= 3:
            long_paragraphs += 1
    if long_paragraphs:
        warnings.append(f"3文以上の段落が {long_paragraphs} 箇所(1〜2文推奨)")

    return warnings


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
    kwargs: dict = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "system": render_system_prompt(args.products, iso_date),
        "messages": [
            {
                "role": "user",
                "content": render_user_prompt(args.target, args.products, iso_date),
            }
        ],
    }
    # effort は Claude 5 系のみ。Haiku 4.5 等に送ると 400 になるので付けない。
    if EFFORT_CAPABLE.search(args.model):
        kwargs["output_config"] = {"effort": args.effort}

    # 長文生成のためストリーミングで受ける(非ストリーミングは HTTP タイムアウトの恐れ)。
    with client.messages.stream(**kwargs) as stream:
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
        f"  model={message.model} in={usage.input_tokens} out={usage.output_tokens}",
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
        description="Claude API で Hugo 用の DTM 機材レビュー記事を生成する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default=DEFAULT_TARGET, help="機材名またはテーマ")
    parser.add_argument(
        "--products", type=int, default=DEFAULT_PRODUCTS, help="個別セクションを作る製品数"
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
    parser.add_argument(
        "--placeholders",
        default="comment",
        choices=["comment", "raw"],
        help="画像/動画プレースホルダの扱い。comment=TODOコメント化, raw=そのまま出力",
    )
    parser.add_argument("--slug", help="ファイル名の slug を明示指定する(省略時は自動)")
    parser.add_argument("--outdir", type=Path, default=POSTS_DIR, help="出力先ディレクトリ")
    parser.add_argument(
        "--force", action="store_true", help="同名ファイルが存在する場合に上書きする"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ファイルに保存せず標準出力に表示する(API 呼び出しは行う)",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="組み立てたプロンプトを表示して終了する(API 呼び出しなし・課金なし)",
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

    if args.print_prompt:
        system = render_system_prompt(args.products, iso_date)
        user = render_user_prompt(args.target, args.products, iso_date)
        print("=" * 70)
        print(f"SYSTEM PROMPT ({len(system)} 文字)")
        print("=" * 70)
        print(system)
        print("=" * 70)
        print(f"USER PROMPT ({len(user)} 文字)")
        print("=" * 70)
        print(user)
        effort = "有効" if EFFORT_CAPABLE.search(args.model) else "非対応のため送信しない"
        print(
            f"model={args.model}  max_tokens={args.max_tokens}  "
            f"effort={args.effort} ({effort})"
        )
        return 0

    client = build_client()

    print(
        f"生成中: target={args.target!r} products={args.products} "
        f"model={args.model} effort={args.effort}"
    )
    markdown = generate(client, args, iso_date)
    markdown = force_date(markdown, iso_date)
    markdown = fix_affiliate_urls(markdown)

    if not markdown.startswith("---"):
        sys.exit("出力が Front Matter (---) で始まっていません。プロンプトを確認してください。")

    images = videos = 0
    if args.placeholders == "comment":
        markdown, images, videos = neutralize_media(markdown)

    warnings = lint(markdown, args.products)

    if args.dry_run:
        print(markdown)
    else:
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

    if args.placeholders == "comment" and (images or videos):
        print(f"  プレースホルダ: 画像 {images} 件 / 動画 {videos} 件 を TODO コメント化")

    if warnings:
        print("\n[品質チェック]", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)
    else:
        print("  品質チェック: 問題なし")

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
