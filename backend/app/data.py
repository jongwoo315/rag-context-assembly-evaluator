SAMPLE_CORPUS = [
    {
        "id": "doc_1",
        "source": "billing",
        "text": "2024년 회원 요금제는 다음과 같습니다. \n\n배송 요금은 기본 3,000원에 배달 거리에 따라 추가 요금이 붙습니다. \n\nVIP 회원은 배달 요금을 50% 할인받을 수 있습니다.",
        "metadata": {"category": "billing"},
    },
    {
        "id": "doc_2",
        "source": "store",
        "text": "식당 운영 시간은 오전 10시부터 오후 10시까지입니다. \n\n주말에는 부션 운영을 합니다. \n\n배달 서비스는 운영 종료 30분 전까지 주문을 받습니다.",
        "metadata": {"category": "store"},
    },
    {
        "id": "doc_3",
        "source": "cs",
        "text": "드립 커피 이슈 시 기기 청소를 먼저 진행해 주세요. \n\n그리지 않으면 수도 배관을 확인한 후 교환을 요청하세요. \n\n교환 건은 구매 후 7일 이내에 처리됩니다.",
        "metadata": {"category": "cs"},
    },
    {
        "id": "doc_4",
        "source": "billing",
        "text": "추가 배달 요금은 거리당 500원입니다. \n\nVIP 회원의 할인율은 전 요금에 자동 적용되며, 결제 후 영수증을 발급합니다.",
        "metadata": {"category": "billing"},
    },
]

SAMPLE_QUESTIONS = [
    {
        "query": "VIP 회원의 배달 요금을 알려줘.",
        "expected_doc_id": "doc_1",
        "expected_chunk_ids": ["doc_1_c2"],
    },
    {
        "query": "식당 운영 시간이 어떻게 돼?",
        "expected_doc_id": "doc_2",
        "expected_chunk_ids": ["doc_2_c0"],
    },
    {
        "query": "드립 커피 기기 문제 시 어떻게 처리해?",
        "expected_doc_id": "doc_3",
        "expected_chunk_ids": ["doc_3_c0", "doc_3_c1"],
    },
    {
        "query": "추가 배달 요금은 얼마야?",
        "expected_doc_id": "doc_4",
        "expected_chunk_ids": ["doc_4_c0"],
    },
]
