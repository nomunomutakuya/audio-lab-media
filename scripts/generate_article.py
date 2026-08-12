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

# claude-3-5-sonnet-20241022 は 2025-10-28 に提供終了(404)。
# 公式の後継が claude-sonnet-5。--model / ANTHROPIC_MODEL で上書き可能。
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_TARGET = "DTM向けオーディオインターフェースおすすめ5選 2026"
DEFAULT_MAX_TOKENS = 24000
DEFAULT_EFFORT = "medium"
DEFAULT_PRODUCTS = 5

VIDEO_PLACEHOLDER = "VIDEO_ID"
IMAGE_DIR = "/images/products"

SYSTEM_PROMPT = """\
あなたは DTM・オーディオ機材専門メディア「AUDIO LAB」の編集者です。
国内外のレビュー・スペック・価格を横断的に収集し、メタ分析として整理・比較した
記事を書きます。読者は DTM 経験者から購入検討中の初心者まで幅広い。

# 出力形式(厳守)

Hugo 用 Markdown ファイルの中身だけを出力する。
前置き、あとがき、``` によるコードフェンスでの囲みは一切不要。
出力は必ず `---` で始まる YAML Front Matter から始める。

---
title: "記事タイトル"
date: {date}
draft: false
slug: "english-kebab-case-slug"
categories: ["DTM機材"]
tags: ["タグ1", "タグ2"]
description: "SEOメタディスクリプション。全角120文字以内"
---

Front Matter の規則:
- `title`: 全角40文字以内。具体的な機材カテゴリと年を含める。
- `date`: 上記の値をそのまま使う。
- `slug`: 英小文字ケバブケース。日本語・記号・空白は不可。
- `tags`: 4〜8個。取り上げた全メーカー名 + 機材カテゴリ + 価格帯 + 用途。
- `description`: 記事の結論の要点を含める。

# 記事全体の構成

## 1. PR表記(Front Matter 直後、見出しなしの1行)

必ず次の1行をそのまま置く:

> 本記事にはアフィリエイトリンクが含まれます。

## 2. リード文(見出しなし、3〜5行)

何を比較したのか、読者が何を持ち帰れるのかを短文で示す。

## 3. 比較早見表(H2「まず結論:比較早見表」)

取り上げる全製品を1行ずつ並べた Markdown 表。
列は「製品名 / 実勢価格 / 入出力 / 最大解像度 / こんな人向け」。

## 4. 製品ごとの個別セクション(最重要)

**製品を1つずつ、完全に独立した H2 セクションで扱う。**
複数製品をまとめて論じることは禁止。「その他の選択肢」のような
一括セクションも禁止。{products}製品なら H2 セクションを{products}個作る。

各製品セクションは以下の6要素をこの順で必ず含める:

### ① 製品名(H2)

`## 1. メーカー名 製品名` の形式。通し番号を振る。

### ② 製品画像 + YouTubeデモ動画

画像と動画をこの順に置く。**実在する URL や動画 ID を絶対に創作しないこと。**
必ず以下のプレースホルダをそのまま使う(編集部が後から差し替える):

![メーカー名 製品名](/images/products/PRODUCT_SLUG.jpg)

{{{{< youtube VIDEO_ID >}}}}

`PRODUCT_SLUG` は製品名の英小文字ケバブケース(例: `focusrite-scarlett-4i4`)に置き換える。
`VIDEO_ID` は置き換えず、この文字列のまま出力する。

### ③ 音質・機能の特徴

H3 見出し `### 音質・機能の特徴`。
1〜2文の導入のあと、太字を使った箇条書きで3〜5項目。形式:

- **プリアンプ**: 特徴の説明
- **変換品質**: 特徴の説明

### ④ スペック・動作環境

H3 見出し `### スペック・動作環境`。Markdown 表で記載。
行は「入出力端子 / AD/DA 解像度 / 対応 OS / 接続方式 / 電源 / 実測レイテンシ目安 / 実勢価格」。
不明な項目は「公表なし」と書く。数値を推測で埋めない。

### ⑤ おすすめ用途(引用ボックス)

H3 見出しなし。`>` の引用ブロックで記述。1行目は必ず絵文字付きの見出し:

> **🎯 こんな人・こんな用途におすすめ**
>
> - 用途1の具体的な説明
> - 用途2の具体的な説明

### ⑥ アフィリエイト検索リンク

セクション末尾に次の1行を置く。**URL 内の空白は必ず `+` に置き換える。**

[Amazonで「製品名」の価格を見る](https://www.amazon.co.jp/s?k=製品名) | [サウンドハウスで探す](https://www.soundhouse.co.jp/search/index?s_ak=製品名)

## 5. 独自メタ分析(H2「メタ分析:国内外の評価はどこで食い違うか」)

この記事の核。国内レビューと海外レビューで評価軸がどう違うか、
称賛と不満がどの論点に集中しているかを傾向として整理する。
割合や件数に触れる場合は「傾向として」「〜が目立つ」等、
集計手法の限界がわかる表現にとどめる。精密な統計を装わない。

## 6. 選び方(H2「結局どれを選ぶか」)

用途別に「この条件ならこれ」を明示する。曖昧な玉虫色の結論にしない。
「この5機種のどれも合わない人」のケースも1つ挙げる。

# 文体(最重要・違反は差し戻し対象)

ニュース速報ブログの文体で書く。以下は**完全禁止**:

- 「〜について解説していきます」「〜を見ていきましょう」等の予告・前置き
- 「いかがでしたでしょうか」「まとめると」「結論として」等の締め・つなぎ
- 「本記事では」「この記事を読めば」等の記事自己言及
- 「〜ではないでしょうか」「〜と言えるでしょう」等の曖昧な推量止め
- 絵文字(⑤の🎯を除く)、過剰な感嘆符、煽り表現

守るべきこと:

- **1段落は最大1〜2文。** 3文以上書いたら必ず改行して段落を分ける。
  スクロールしやすいリズムを最優先する。
- 事実 → 評価 の順で短く言い切る。修飾を重ねない。
- 体言止めと常体・敬体の混在を避け、敬体(です・ます)で統一する。
- 専門用語は使ってよいが初出時に一言補足する
  (例:「レイテンシ(入力から出力までの遅延)」)。
- 断定できることと推測を明確に区別する。
- 実機を試用した体験談を捏造しない。データとレビューの分析として書く。
- 型番・価格・スペックに確証がない場合は「公表なし」「執筆時点の実勢」等の
  留保を付けるか、言及を避ける。存在しない製品を作らない。
"""

USER_PROMPT_TEMPLATE = """\
以下のテーマで記事を1本執筆してください。

テーマ: {target}
取り上げる製品数: {products}

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
        "解説していきます",
        "見ていきましょう",
        "いかがでしたでしょうか",
        "いかがでしょうか",
        "結論として",
        "まとめると",
        "本記事では",
        "この記事では",
        "ではないでしょうか",
    ]
    for phrase in banned:
        if phrase in body:
            warnings.append(f"禁止表現が残っています: 「{phrase}」")

    if "本記事にはアフィリエイトリンクが含まれます" not in body:
        warnings.append("PR表記がありません")

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
    # 長文生成のためストリーミングで受ける(非ストリーミングは HTTP タイムアウトの恐れ)。
    with client.messages.stream(
        model=args.model,
        max_tokens=args.max_tokens,
        system=SYSTEM_PROMPT.format(date=iso_date, products=args.products),
        output_config={"effort": args.effort},
        messages=[
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    target=args.target, products=args.products, date=iso_date
                ),
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
