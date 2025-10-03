yago4.5:
	wget https://yago-knowledge.org/data/yago4.5/yago-4.5.0.2.zip -O yago4.5_full.zip
	mkdir yago4.5_full
	unzip yago4.5_full.zip -d yago4.5_full
	rm yago4.5_full.zip
	python strip_yago.py --input-dir yago4.5_full --output-dir yago4.5
	rm -rf yago4.5_full

data/yago4.5: yago4.5
	python yago2dataset.py\
		--input-dir ./yago4.5\
		--output-dir ./data/yago4.5\
		--linearize\
		--sparsity-filter-threshold 1\
		--min-year 1950\
		--max-year 2024\
		--relations\
		schema:memberOf schema:worksFor schema:award yago:academicDegree yago:replaces\
		yago:playsIn schema:location schema:spouse schema:sponsor schema:nationality\
		schema:homeLocation schema:manufacturer yago:ownedBy yago:leader yago:participant\
		schema:contentLocation schema:owns yago:notableWork yago:neighbors yago:capital\
		yago:terminus yago:beliefSystem schema:founder yago:doctoralAdvisor schema:actor\
		yago:sportNumber schema:locationCreated schema:performer schema:musicBy\
		yago:director yago:appearsIn schema:publisher yago:conferredBy yago:candidateIn\
		schema:editor schema:superEvent schema:director schema:inLanguage schema:lyricist\
		schema:organizer schema:material schema:productionCompany yago:studentOf\
		schema:children yago:follows schema:knowsLanguage yago:influencedBy schema:gender\
		schema:recordLabel yago:flowsInto yago:officialLanguage yago:lowestPoint\
		--start-only-relations schema:children schema:founder schema:locationCreated\
		yago:follows yago:influencedBy schema:knowsLanguage schema:lyricist

output/yago4.5/r1,2,3_n200_exp_s12_rules.json: data/yago4.5
	python -m fiction.tlogic.learn\
		--dataset yago4.5\
		--rule_lengths 1 2 3\
		--num_walks 200\
		--seed 12\
		--num_processes 1

output/yago4.5/r1,2,3_n200_exp_s12_cands_r[1,2,3]_w2048_score_12[0.1,0.5].json: output/yago4.5/r1,2,3_n200_exp_s12_rules.json:
	python -m fiction.tlogic.apply\
		--dataset yago4.5\
		--rules r1,2,3_n200_exp_s12_rules.json\
		--rule_lengths 1 2 3\
		--window 2048\
		--num_processes 4

.PHONY: evaluate_tlogic
evaluate_tlogic: output/yago4.5/r1,2,3_n200_exp_s12_cands_r[1,2,3]_w2048_score_12[0.1,0.5].json
	python -m fiction.tlogic.evaluate\
		--dataset yago4.5\
		--candidates r1,2,3_n200_exp_s12_cands_r[1,2,3]_w2048_score_12[0.1,0.5].json

output/yago2026-facts.txt: output/yago4.5/r1,2,3_n200_exp_s12_rules.json
	python -m fiction.gen_new_facts\
		--rules ./output/yago4.5-small/r1,2,3_n200_exp_s12_rules.json\
		--rule-lengths 1 2 3\
		--dataset-dir ./data/yago4.5\
		--yago-dir ./yago4.5\
		--year 2026\
		--mimic-year 2022\
		--process-nb 8\
		--max-queries 4\
		--non-exclusive-relations startAward startNotableWork startNeighbors startFounder startDoctoralAdvisor startPerformer startMusicBy startAppearsIn startConferredBy startCandidateIn\
		--output-file "./output/yago2026-facts.txt"

output/yago2022-facts.txt: data/yago4.5
	python -m fiction.filter_facts\
		--dataset-dir ./data/yago4.5\
		--output-file ./output/yago2022-facts.txt\
		--min-year 2022\
		--max-year 2022

output/yago2026.json: output/yago2026-facts.txt
	python -m fiction.gen_description\
		--facts-file "./output/yago2026-facts.txt"\
		--language-model "hf:meta-llama/Meta-Llama-3.1-8B-Instruct"\
		--output-file "./output/yago2026.json"

output/yago2026_multi.json: output/yago2026-facts.txt yago4.5
	python -m fiction.gen_description\
		--facts-file "./output/yago2026-facts.txt"\
		--language-model "hf:meta-llama/Meta-Llama-3.1-8B-Instruct"\
		--multi-min-size 2\
		--multi-max-size 4\
		--multi-yago-dir "./yago4.5"\
		--multi-alpha 0.9\
		--multi-k 0.03\
		--output-file "./output/yago2026_multi.json"

output/yago2022.json: output/yago2022-facts.txt
	python -m fiction.gen_description\
		--facts-file "./output/yago2022-facts.txt"\
		--language-model "hf:meta-llama/Meta-Llama-3.1-8B-Instruct"\
		--output-file "./output/yago2022.json"

output/yago2022_multi.json: output/yago2022-facts.txt yago4.5
	python -m fiction.gen_description\
		--facts-file "./output/yago2022-facts.txt"\
		--language-model "hf:meta-llama/Meta-Llama-3.1-8B-Instruct"\
		--multi-min-size 2\
		--multi-max-size 4\
		--multi-yago-dir "./yago4.5"\
		--multi-alpha 0.9\
		--multi-k 0.03\
		--output-file "./output/yago2022_multi.json"
