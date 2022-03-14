Math-solver with TUNiB & AIFFEL
=

## KMWP (Korean Math Word Problem)
본 레포지토리는 KMWP부문에서 수학문제를 해결하기 위해 사용된 모델들을 huggingface와 PyTorch를 이용하여, 구현한 공간입니다.
시도해본 모델들은 아래와 같습니다.

## Model




|순번|사용 모델|validation-accuracy|비고|
|---:|---:|---:|---:|
|1|Ko-GPT2|0.54||
|2|T-5|0.27|낮은 성능으로 실험 중단|
|3|encoder-decoder(kobert-kobert)|0.27|낮은 성능으로 실험 중단|
|4|seq2tree|0.66||
|5|verifier|0.22|실험 중|


