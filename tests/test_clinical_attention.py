from clinical_knowledge.clinical_attention import build_clinical_attention


def test_attention_red_flags_high():
    items = build_clinical_attention(
        query="кашель",
        proto_list=[{"path": "x/adult.pdf", "title": "взр_нас", "confidence_score": 0.9}],
        red_flags=["Госпitalизация при нарастании одышки."],
        audience_inferred="adult",
    )
    assert items
    assert items[0]["severity"] == "high"


def test_attention_pregnancy_pediatric_mismatch():
    items = build_clinical_attention(
        query="беременность\nКонтекст подбора: беременные",
        proto_list=[
            {
                "path": "minzdrav/nevrologiya/kp_детс_нас.pdf",
                "title": "детс нас",
                "confidence_score": 0.8,
            }
        ],
        red_flags=[],
        audience_inferred="adult",
    )
    texts = " ".join(i["text"] for i in items)
    assert "беремен" in texts.lower() or "акушер" in texts.lower() or "детск" in texts.lower()
