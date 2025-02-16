## Notebook for keeping track of progress




#### Question Type Annotation

definitions = {
  'POLAR': "A question that can only be answered with 'yes' or 'no'.",
  'ALTERNATIVE': "A question presenting exactly two or three clear options to choose from.",
  'MENTION_SOME': "A question asking for a few examples or instances, not requiring all  possible answers.",
  'MENTION_ALL': "A question requiring a complete list of all possible answers.",
  'SINGLE_MATCH': "A question with exactly one correct answer.",
  'MULTIPLE_WH': "A question using multiple wh-words (who, what, where, when, why, how).",
}


### TyDiQA - GoldP - dataset overview

TyDi QA is a question answering dataset covering 11 typologically diverse languages with 204K question-answer pairs. To provide a realistic information-seeking task and avoid priming effects, questions are written by people who want to know the answer, but don’t know the answer yet, (unlike SQuAD and its descendents) and the data is collected directly in each language without the use of translation (unlike MLQA and XQuAD).

```
This is an example of a validation split instance:

    "annotations": {
        "minimal_answers_end_byte": [-1, -1, -1],
        "minimal_answers_start_byte": [-1, -1, -1],
        "passage_answer_candidate_index": [-1, -1, -1],
        "yes_no_answer": ["NONE", "NONE", "NONE"]
    },
    "document_plaintext": "\"\\nรองศาสตราจารย์[1] หม่อมราชวงศ์สุขุมพันธุ์ บริพัตร  (22 กันยายน 2495 -) ผู้ว่าราชการกรุงเทพมหานครคนที่ 15 อดีตรองหัวหน้าพรรคปร...",
    "document_title": "หม่อมราชวงศ์สุขุมพันธุ์ บริพัตร",
    "document_url": "\"https://th.wikipedia.org/wiki/%E0%B8%AB%E0%B8%A1%E0%B9%88%E0%B8%AD%E0%B8%A1%E0%B8%A3%E0%B8%B2%E0%B8%8A%E0%B8%A7%E0%B8%87%E0%B8%...",
    "language": "thai",
    "passage_answer_candidates": "{\"plaintext_end_byte\": [494, 1779, 2931, 3904, 4506, 5588, 6383, 7122, 8224, 9375, 10473, 12563, 15134, 17765, 19863, 21902, 229...",
    "question_text": "\"หม่อมราชวงศ์สุขุมพันธุ์ บริพัตร เรียนจบจากที่ไหน ?\"..."
}
```

>[note]
TyDi QA has a helpful annotation that lets us easily split the data into questions that can be answered with a yes or no question, and questions that cannot.



### Dataset processing
see tidy-filter.py







|Language | polar count - valid | ~ polar count - valid | polar count - train | ~ polar count train |
|-----|----|-----| ---| ----|
|English|77|954|444|8767|
|Russian|195|1430|994|11809|
|Japanese|142|1567|1031|15257|
|Arabic|98|1282|1179|21913|
|Finnish|94|1988|950|14335|
|Korean|65|1633|296|10685|
|Indonesian|64|1741|328|14624|


