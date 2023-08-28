from os import listdir, environ
import json
import os
import subprocess
import time
import psutil

def get_avg_intervals(samples):
	p75 = 0
	# sort the array in ascending order and get the 75th percentile
	samples.sort()
	p75_arr = samples[int(len(samples) * 0.75):]
	p75 = sum(p75_arr) / len(p75_arr)

	p25 = 0
	# sort the array in ascending order and get the 25th percentile
	samples.sort()
	p25_arr = samples[:int(len(samples) * 0.25)]
	p25 = sum(p25_arr) / len(p25_arr)

	avg = 0
	# get the average of the array
	avg = sum(samples) / len(samples)

	return {
		"p75": p75,
		"p25": p25,
		"avg": avg,
	}

def dig_to_subscript(string):
	s = ['₀','₁','₂','₃','₄','₅','₆','₇','₈','₉']
	return '.'.join(''.join(s[int(i)] for i in string) for string in string.split('.'))

def append_result_markdown_table(name, results):
	# find min, max, and median for each key
	mins = {}
	maxs = {}
	meds = {}

	for result in results:
		if result is None:
			continue

		for key, value in result.items():
			if key not in mins:
				mins[key] = value
				maxs[key] = value
				meds[key] = []
			mins[key] = min(mins[key], value)
			maxs[key] = max(maxs[key], value)
			meds[key].append(value)
	for key, value in meds.items():
		meds[key] = sorted(value)[len(value) // 2]

	# generate markdown table
	md = '\n'
	md += f"## {name}\n"

	# get first non-None result
	first_result = None
	for result in results:
		if result is not None:
			first_result = result
			break
	
	if first_result is None or len(meds) == 0:
		return md + "No results. All runs failed.\n"
	
	keys = first_result.keys()

	# table header
	md += "| " + " | ".join(keys) + " |\n"
	md += "| " + " | ".join(["---"] * len(keys)) + " |\n"

	# table body, median ± half range
	for key in keys:
		md += f"**{str(round(meds[key], 3))}±{dig_to_subscript(str(round(abs(mins[key] - maxs[key]) / 2, 3)))}** | "
	
	md += "\n"

	# table body, individual results
	for result in results:
		for key in keys:
			if result is None:
				md += f"*DNF* | "
			else:
				md += f"{str(round(result[key], 3))} | "
		md += "\n"
	return md

if __name__ == "__main__":
	runs = 0
	to_run = int(environ.get("MIRAMARK_RUNS", 10))
	results = {}

	while runs < to_run:
		print(f"Run {runs + 1}/{to_run} ", end="", flush=True)
		for benchmark in sorted(listdir("suite")):
			# skip files that start with an underscore
			if benchmark.startswith("_"):
				continue
			time.sleep(3) # wait a bit to let the system settle
			mem_baseline = psutil.virtual_memory().used / 1024 / 1024
			proc = subprocess.Popen(["python3", f"suite/{benchmark}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			samples = []
			# take cpu usage and memory usage samples every seconds
			while proc.poll() is None:
				samples.append({
					"cpu": psutil.cpu_percent(interval=1),
					"memory": max(0, psutil.virtual_memory().used / 1024 / 1024 - mem_baseline),
				})

			if benchmark not in results:
				results[benchmark] = []
				
			if proc.returncode != 0 or proc.stdout is None:
				results[benchmark].append(None)
				print("x", end="", flush=True)
			try:
				# each benchmark should output a json object containing numbers for each metric it tracks as the last line
				# ex: {"dataset_processing_time": 0.123, "fit_time": 0.456}
				output = json.loads(proc.stdout.read().decode().splitlines()[-1])
				cpu = get_avg_intervals([sample['cpu'] for sample in samples])
				memory = get_avg_intervals([sample['memory'] for sample in samples])
				results[benchmark].append({
					"cpu_avg": cpu['avg'],
					"cpu_p75": cpu['p75'],
					"cpu_p25": cpu['p25'],
					"memory_avg": memory['avg'],
					"memory_p75": memory['p75'],
					"memory_p25": memory['p25'],
					**output
				})
				print(".", end="", flush=True)
			except:
				# failed to parse json, toss out the result
				results[benchmark].append(None)
				print("x", end="", flush=True)
		print(" ✓", flush=True)

		runs += 1

	# save results to results.md and historic/{ran_at}.md
	ran_at = time.strftime('%Y-%m-%d %H:%M:%S')
	results_md = f"# MiraMark Results\n"
	results_md += f"Ran `{to_run}` times at `{ran_at}` on `{os.uname().nodename}`\n"
	for benchmark, suite_results in results.items():
		results_md += append_result_markdown_table(benchmark, suite_results)

	with open("results.md", "w") as f:
		f.write(results_md)
	
	if not os.path.exists("historic"):
		os.mkdir("historic")

	with open(f"historic/{ran_at}.md", "w") as f:
		f.write(results_md)

	print("Done!")
