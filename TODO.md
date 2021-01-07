% transfer learning between different architectures via weights injection
% [[TLI -> Transfer Learning by Injection]]

[RELATED]

"net2net"
- https://arxiv.org/pdf/1511.05641.pdf
- (analiza!) https://github.com/abhshkdz/papers/blob/master/reviews/net2net-accelerating-learning-via-knowledge-transfer.md
- https://github.com/erogol/Net2Net
- (analiza) https://github.com/erogol/Net2Net/blob/master/net2net.py
-
"FNA++"
- https://arxiv.org/pdf/2006.12986.pdf

"Transferring Knowledge" / "Leap"
- https://openreview.net/pdf?id=HygBZnRctX

[PLAN]

% template: JMLR, ICLR, NeurIPS, ICML, ACL, CVPR

~10 styczen:
	poprawki wszystkich -> prawie finalna wersja
	a) poprawa + dokumentacja kodu
	b) slownictwo / cytowania -> optymalizacja
	c) reformat latex-a (online coding + gmeet)
~30 styczen:
	wrzuta + ostatnie szliwy
	a) stworzenie tutorial-i na Kaggle

[TODO] ;-)

- [ ] wersja pracy w wersji [prezentacji]
		(1) slajd porownanie [net2net | FNA++ | our]
		(2) przyklad uzycia [reset params (przekreslone) | nasze]
- [ ] wersja pracy w wersji [posteru]
- [ ] "stronka pracy" jak https://mtl.yyliu.net/
- [ ] przeczyscic kod --> TODO/FIXME w jednym miejscu (usunac README.md)
- [ ] przeniesc kopie Google Slides (TLI: figures) do repo

- [ ] slownictwo! nie transfer learning -> tylko fine-tuning
- [ ] ComboInjection jako zlozenie paru operacji:
	a) resize         - lambda
	b) center crop    - 1 - lambda
	c) overlap param (np. czy resize - center crop 'sie nachodza")
	d) ? --> max abs / entropy gain
	y = a * CenterCrop(x) + (1 - a) * Resize(x - CenterCrop(x) * b)
- [ ] "WS" --> RYSUNKI jak tutaj https://arxiv.org/pdf/1903.10520.pdf
		albo tutaj https://arxiv.org/pdf/2012.08859.pdf

- [ ] "style of writing" https://www.gwern.net/GPT-3#
- [ ] do "Results and analysis" dodac "potezna" tabelke
- [ ] analyze (https://github.com/goodfeli/dlbook_notation/)
	powolac sie tutaj: "Transfer learning What and how to transfer are key issues
to be addressed in transfer learning"
	---> https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Meta-Transfer_Learning_for_Few-Shot_Learning_CVPR_2019_paper.pdf
- [ ] !!! dodac do apply_TTA --> apply_reset przed
- [ ] (figure at first page) https://openreview.net/pdf?id=rJxAo2VYwr
- [ ] zrobic skrypt to automatycznego podmieniania font-a na latex w *.pdf
	--> cos ala inscape / ghostscript / latex
- [X] pokazac 'use case' + link do repozytorium
- [ ] KOLORY [REF]: w pracy na niebiesko!!! (wtedy dobrze wyglada)
- [ ] analiza -> podzial pracy: https://arxiv.org/pdf/1904.04232.pdf
- [ ] zrobic figure na pierwsza strone:
	a) prosty i malych -> tylko 3 linie (albo bar plot) jak w WS
	b) tlumaczycy metode -> diagram? [1. matching / 2. remapping]
		(taki receptor jak z pracy o FNA++)
- [ ] analiza: https://mtl.yyliu.net/
	a) czy tez zrobic proste wykresy (czerwony / niebieski / czarny)
	     "na pierwsza strone" -> tylko "TLI(imagenet)" / init / TF (dotted)
	b) jak zrobili ten figure "scaling i shifting" (google slides?)
- [ ] ladnie zapisac wniosek:
	"jakakolwiek wiedza jest lepsza niz brak wiedzy"
- [X] uczyc na dwoch zbiorach - "MNIST-55K" / "CIFAR100"
- [ ] ujednolicic nazewnictwo "number of steps" zamiast "iterations"
	(DropMax) https://arxiv.org/pdf/1712.07834.pdf
	"Base test" itc.
- [ ] exp3: trzeba powtorzyc jak zmieni sie algo (lub zostanie batchnorm?)
- [ ] exp?: zweryfikowac czy przenoszenie parametrow optymalizatora
             --> pomaga w nauce
- [ ] przemyslec oznaczenia na wykresach
		Q1: uzywac notacji ze strzalka? "->"
		Q2: a moze TLI(from=?, to=?)
- [ ] pokazac ze KD na pewnych warstwach wraz z treningiem dazy do
	"przeniesionych" wag (losowy init -> rozklad)
	a) wykonac dwa rysunki (o co kaman)
	b) na malym przykladzie MNIST-a zwizualicowac
	c) czy mozna dac w pracy ladne wzory / matematyke / dowod?
- [ ] dodac do wykresu "tego co bedzie na pierwszej stronie?"
-		legende co oznacza TLI(score=0.1)
-			-> jakie modele zostaly uzyte
- [ ] analiza: https://arxiv.org/pdf/1702.06295.pdf
- [ ] czy jest zaleznosc pomiedzy: tli score a FLOPS
- [ ] tli score -> (model -> all) vs. (all -> model)
- [ ] analiza sekcji "features" w:
	https://github.com/rwightman/pytorch-image-models
- [ ] w tli.py:
	a) dokonczyc wizualizacje dystrybucji oraz kullbeck-a
- [ ] sprawdzic inne optimizery? smooth_labeling? --> [SUPER DUPER FAST]
- [ ] lepsza generacja sieci w `get_model_debug`
- [ ] score diff -> best `tli.py` copied to `tli_best/` path
- [ ] faster code? pytorch? vectorize? multi-thread?
- [ ] debug dashboard -> szybki test czy poprawiamy algo czy tez nie
- [ ] zrobic skrypty takie jak: scripts/cifar10.sh scripts/format.sh
		powtarzalnosc eksperymentow
- [ ] inspiracja:
	https://arxiv.org/pdf/1912.12522.pdf
	https://iclr.cc/virtual_2020/poster_H1loF2NFwr.html
	https://github.com/kcyu2014/eval-nas
	https://github.com/D-X-Y/Awesome-AutoDL
- [ ] czy przenoszenie wag mozna augmentowac?
	{scale, mixup(a, init), ???, cutout, gridaug, ricap}

[SVG]

- [ ] na plot-cie zaznaczyc "nazwy" modeli (w srodku przy punktach/liniach)
- [ ] automatyczny system przeksztalcania svg/pdf to formatu latex
	--> pdf-crop-margins -v -s -u <file>

pdfCropMargins
https://inkscape.org/~Xaviju/%E2%98%85tools-organization
textext

[REFS]

- analyze: https://github.com/VDIGPKU/DADA (table)
- analyze (vis):
     https://github.com/human-analysis/neural-architecture-transfer
- zbiory: CIFAR-10 / CIFAR-100 / SVHN / Reduced ImageNet

David Lynch: https://www.youtube.com/watch?v=gM8dUPHv2HY

--> scoring function: https://www.youtube.com/watch?v=a6v92P0EbJc&t=1s
--> experimental setup: https://arxiv.org/pdf/1904.04232.pdf
--> options: gwern: GPT-3 article

- "Distilling Optimal Neural Networks: Rapid Search in Diverse Spaces"
https://arxiv.org/pdf/2012.08859.pdf
(ladna figury -- nasladowac)

"Knowledge Distillation: A Survey"
- https://arxiv.org/pdf/2006.05525.pdf
- https://www.youtube.com/results?search_query=language+in+scientific+papers+english
- DropHead: github repo

% FIXME: google
% --> "Graph similarity scoring and matching"
% --> "similarity between two graphs"

- https://www.buildgpt3.com/ (magicflow?)
- https://transformer.huggingface.co/
- https://app.wordtune.com/playground
- https://www.wordtune.com/ (https://www.ai21.com/)
- https://demo.allennlp.org/masked-lm?text=The%20%5BMASK%5D%20%5BMASK%5D%20to%20the%20emergency%20room%20to%20see%20his%20patient.
- https://www.reddit.com/r/GPT3/comments/iurj2q/gpt3_alternative/
- https://play.aidungeon.io/main/play?publicId=f0f90ac7-3c9f-4641-82ba-8f3faf545e54

- https://github.com/emptymalei/awesome-research
- https://github.com/Wookai/paper-tips-and-tricks

- https://github.com/joe-siyuan-qiao/WeightStandardization
- https://arxiv.org/pdf/2012.08859.pdf
- https://arxiv.org/pdf/1903.10520.pdf
- https://mtl.yyliu.net/
- https://arxiv.org/pdf/2009.09152.pdf
- https://arxiv.org/pdf/1712.07834.pdf
- https://arxiv.org/pdf/2004.07636.pdf

ADNOTACJE NA WYKRESACH JAKBY BYL TO "GOOGLE SLIDES"
------------------> czyli strzalki z podpisami

https://github.com/brenhinkeller/preprint-template.tex
