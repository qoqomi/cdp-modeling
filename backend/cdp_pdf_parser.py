"""
CDP Questionnaire PDF to JSON Parser
=====================================
CDP 질문지 PDF를 구조화된 JSON으로 변환하는 파서

사용 방법:
    python cdp_pdf_parser.py --input data/Full_Corporate_Questionnaire.pdf --output data/cdp_questions.json

Note: 이 파일은 backward compatibility를 위한 래퍼입니다.
      실제 로직은 model/questionnaire.py에 있습니다.
"""

from model import QuestionnaireParser
from model.questionnaire import (
    FieldType,
    ResponseColumn,
    Tag,
    CDPQuestion,
)


class CDPPDFParser(QuestionnaireParser):
    """
    CDP PDF Parser (Backward Compatible Wrapper)

    기존 코드와의 호환성을 위해 유지됩니다.
    새 코드에서는 model.QuestionnaireParser를 직접 사용하세요.
    """

    def parse_all(self):
        """기존 메서드명 유지 (parse()로 위임)"""
        return self.parse()


# TableExtractor는 선택적 기능으로 별도 유지
try:
    from pdf2image import convert_from_path
    from PIL import Image
    import torch
    from transformers import AutoModelForObjectDetection, AutoImageProcessor

    HAS_TABLE_DETECTION = True
except ImportError:
    HAS_TABLE_DETECTION = False


class TableExtractor:
    """표 추출기 (Table Transformer 사용) - 선택적 기능"""

    def __init__(self):
        if not HAS_TABLE_DETECTION:
            raise ImportError("Table detection libraries not installed")

        self.processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        self.model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        )

    def extract_tables_from_page(self, image: Image.Image):
        """페이지에서 표 감지 및 추출"""
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes
        )[0]

        tables = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = box.tolist()
            tables.append(
                {
                    "score": score.item(),
                    "label": self.model.config.id2label[label.item()],
                    "box": box,
                }
            )

        return tables


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CDP PDF to JSON Parser")
    parser.add_argument("--input", "-i", required=True, help="Input PDF path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path")

    args = parser.parse_args()

    # PDF 파싱
    print(f"Parsing {args.input}...")
    pdf_parser = CDPPDFParser(args.input)
    questions = pdf_parser.parse_all()

    # JSON 저장
    pdf_parser.to_json(args.output)

    print(f"\nTotal questions parsed: {len(questions)}")


if __name__ == "__main__":
    main()
