from pathlib import Path
import fitz  # PyMuPDF

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/text")

OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_pdf_text(pdf_path: Path) -> str:
    text = []
    doc = fitz.open(pdf_path)
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)

def main():
    for pdf_file in RAW_DIR.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        text = extract_pdf_text(pdf_file)
        out_file = OUT_DIR / (pdf_file.stem + ".txt")
        out_file.write_text(text, encoding="utf-8")

    print("✅ Text extraction complete.")

if __name__ == "__main__":
    main()
