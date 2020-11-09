# Helper functions for submitting jobs to compute nodes on Habanero, Columbia's high-performance computing cluster, which uses Slurm to manage jobs. You can replace this file with one customized for your system; just preserve the signature of the submit_job function. See FrameData.extract_and_pickle_gwfs for an example of usage.
# NB: with Habanero's configuration of Slurm, when batch jobs are submitted they inherit the environment variables of the submitting process, including $PATH. This means that if the process that submits a batch job has a particular Conda environment active, it will effectively be active for the batch job as well. That enables jobs submitted with this module's submit_job function to run with access to all the packages installed in the Conda environment active for the process that called it. If your system works differently, you may have to find another solution.

import subprocess
import time
import datetime
import os
import numpy as np


max_cpus = 24
max_mem = 125
squeue_header_line = "             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)"
node_list = ["node{:03d}".format(n) for n in range(1, 303)]


def call_squeue():
	return subprocess.run(["squeue"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.splitlines()


def call_squeue_with_retry():
	out_lines = call_squeue()
	while len(out_lines) < 1 or out_lines[0] != squeue_header_line:
		# Error running squeue; wait and try again
		time.sleep(5)
		out_lines = call_squeue()
	return out_lines


def call_sbatch(sh_filename):
	return subprocess.run(["sbatch", sh_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def get_idle_nodes():
	out_lines = call_squeue_with_retry()
	running_lines = [line for line in out_lines if line.split()[4] == "R"]
	nodes_in_use = set([line.split()[-1] for line in running_lines])
	for node in nodes_in_use.copy():
		if "[" in node:
			nodes_in_use.remove(node)
			ranges = node.replace("]", "[").split("[")[1].split(",")
			for rg in ranges:
				if "-" in rg:
					range_start, range_end = map(int, rg.split("-"))
				else:
					range_start, range_end = int(rg), int(rg)
				nodes_in_use.update(["node{:03d}".format(n) for n in range(range_start, range_end + 1)])
	idle_nodes = [node for node in node_list if node not in nodes_in_use]
	return idle_nodes


def make_python_command(func_name, args, arg_names=None, imports=None, import_paths=None):
	if arg_names is None:
		arg_names = [None]*len(args)
	if len(arg_names) != len(args):
		raise ValueError

	python_command = ""
	if import_paths is not None:
		python_command += "import sys; "
		import_paths = [import_paths] if isinstance(import_paths, str) else import_paths
		for path in import_paths:
			python_command += "sys.path.insert(1, \"{}\"); ".format(path)
	if imports is not None:
		imports = [imports] if isinstance(imports, str) else imports
		for im in imports:
			python_command += "import {}; ".format(im)

	python_command += "{}(".format(func_name)

	prev_arg_named = False
	for arg_name, arg in zip(arg_names, args):
		if arg_name is not None:
			python_command += "{}=".format(arg_name)
			prev_arg_named = True
		elif prev_arg_named:
			raise ValueError
		if isinstance(arg, str):
			python_command += "\"{}\", ".format(arg)
		else:
			python_command += "{}, ".format(arg)

	python_command += ")"

	return python_command


def write_sh(python_command, job_name, time_hrs, time_mins=0, time_secs=0, num_nodes=1, num_cpus=max_cpus, total_mem=max_mem, gpu=0, exclusive=False, comp=True, pre_python_commands=()):
	if not comp and os.path.splitext(python_command.split()[0])[1] != ".py":
		raise ValueError
	sh_filename = job_name + ".sh"
	with open(sh_filename, "w") as shf:
		shf.write("#!/bin/sh\n")
		shf.write("#\n")
		shf.write("#SBATCH --account=geco\n")
		shf.write("#SBATCH --job-name={}\n".format(job_name))
		shf.write("#SBATCH --nodes={}\n".format(num_nodes))
		shf.write("#SBATCH --cpus-per-task={}\n".format(num_cpus))
		shf.write("#SBATCH --mem={}gb\n".format(total_mem))
		if gpu != 0:
			shf.write("#SBATCH --gres=gpu:{}\n".format(gpu))
		if exclusive:
			shf.write("#SBATCH --exclusive\n")
		shf.write("#SBATCH --time={}:{}:{}\n".format(time_hrs, time_mins, time_secs))
		shf.write("\n")
		for command in pre_python_commands:
			shf.write(command + "\n")
		if comp:
			shf.write("python -c \"{}\"".format(python_command.replace("\"", "\\\"")))
		else:
			shf.write("python {}".format(python_command))
		shf.write("\n")
	return sh_filename


def submit_sh(sh_filename):
	while True:
		out = call_sbatch(sh_filename)
		if not out.stderr:
			job_id = int(out.stdout.split()[-1])
			# Assumes stdout of successful job submission is "Submitted batch job <n>" where <n> is the job id.
			# We keep retrying if stderr is not empty to catch the common "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation". But this could still fail if there's something else unexpected.
			print("Submitted job {}: {}".format(job_id, sh_filename))
			return job_id
		else:
			print("Submitting {} failed; retrying".format(sh_filename))
			time.sleep(5)


def submit_job(python_command, job_name, time_hrs, time_mins=0, time_secs=0, num_nodes=1, num_cpus=max_cpus, total_mem=max_mem, gpu=0, exclusive=False, comp=True, pre_python_commands=()):
	sh_filename = write_sh(
		python_command=python_command,
		job_name=job_name,
		time_hrs=time_hrs,
		time_mins=time_mins,
		time_secs=time_secs,
		num_nodes=num_nodes,
		num_cpus=num_cpus,
		total_mem=total_mem,
		gpu=gpu,
		exclusive=exclusive,
		comp=comp,
		pre_python_commands=pre_python_commands
	)
	job_id = submit_sh(sh_filename)
	return sh_filename, job_id


def check_file_done(path, last_modified=15, wait_if_writing=True):
	if not os.path.isfile(path):
		return False
	else:
		while wait_if_writing:
			if time.time() - os.path.getmtime(path) > last_modified:
				wait_if_writing = False
			else:
				time.sleep(1)
		return True


def check_files_done(paths, wait_if_writing=True):
	# Each element of paths can be a string path or a list (or tuple/set/array) of paths. In the latter case, is_done[i] will be True if and only if all files in paths[i] are done.
	is_done = []
	for path in paths:
		if isinstance(path, (list, tuple, set, np.ndarray)):
			is_done.append(all(check_file_done(f, wait_if_writing) for f in path))
		else:
			is_done.append(check_file_done(path, wait_if_writing))
	return is_done


def wait_for_files(paths, check_interval=15):
	done = False
	while not done:
		files_awaiting = sum(not file_done for file_done in check_files_done(paths, wait_if_writing=True))
		if files_awaiting > 0:
			print("{}: Waiting for {} files...".format(datetime.datetime.now(), files_awaiting))
			time.sleep(check_interval)
		else:
			done = True
	print("{}: Done".format(datetime.datetime.now()))


def check_jobs_status(job_ids):
	if not isinstance(job_ids, (list, tuple, set, np.ndarray)):
		job_ids = [job_ids]
	job_ids = [str(job_id) for job_id in job_ids]

	out_lines = call_squeue_with_retry()
	jobs_in_progress_ids, jobs_in_progress_statuses = [], []
	for line in out_lines[1:]:
		tokens = line.split()
		if tokens[0] in job_ids:
			jobs_in_progress_ids.append(tokens[0])
			jobs_in_progress_statuses.append(tokens[4])

	statuses = []
	for job_id in job_ids:
		if job_id in jobs_in_progress_ids:
			statuses.append(jobs_in_progress_statuses[jobs_in_progress_ids.index(job_id)])
		else:
			statuses.append(None)

	return statuses


def num_jobs_in_progress(job_ids):
	return sum(status is not None for status in check_jobs_status(job_ids))


def wait_for_jobs(job_ids, check_interval=15):
	done = False
	while not done:
		jobs_in_progress = num_jobs_in_progress(job_ids)
		if jobs_in_progress > 0:
			print("{}: Waiting for {} jobs...".format(datetime.datetime.now(), jobs_in_progress))
			time.sleep(check_interval)
		else:
			done = True
	print("{}: Done".format(datetime.datetime.now()))


def check_for_failed_jobs(job_ids, output_paths):
	# A job is considered to have failed if its status is None (i.e. not running or pending) and its output file(s) don't exist
	if len(output_paths) != len(job_ids):
		raise ValueError
	jobs_status = check_jobs_status(job_ids)  # This can be slow. It's possible for jobs to finish and write their output files during/after this but before checking output files in the next line, which is fine. But checking for output files first could result in jobs erroneously being considered to have failed because a job could finish and write its output after that check but before its status is checked.
	jobs_success = check_files_done(output_paths)
	failed_ids = [job_id for job_id, is_success, status in zip(job_ids, jobs_success, jobs_status) if status is None and not is_success]
	return failed_ids


def retry_jobs_if_failed(job_ids, sh_filenames, output_paths):
	# Nothing happens for jobs that are still pending or running. Return value job_ids_new is the same as job_ids except the IDs of any failed jobs are replaced with the IDs of the corresponding resubmitted jobs.
	if len(job_ids) != len(sh_filenames) or len(job_ids) != len(output_paths):
		raise ValueError
	failed_ids = check_for_failed_jobs(job_ids, output_paths)

	retry_ids, job_ids_new = [], []
	for job_id, sh_filename in zip(job_ids, sh_filenames):
		if job_id in failed_ids:
			print("Job {} failed; resubmitting".format(job_id))
			retry_id = submit_sh(sh_filename)
			retry_ids.append(retry_id)
			job_ids_new.append(retry_id)
		else:
			job_ids_new.append(job_id)

	return job_ids_new, failed_ids, retry_ids


def retry_jobs_until_success(job_ids, sh_filenames, output_paths, retry_interval=30):
	# Specify retry_interval <= 0 to wait for all running job_ids to finish before checking for and retrying any failed ones

	all_job_ids = list(job_ids)  # Copy
	done = False
	while not done:
		if retry_interval <= 0:
			wait_for_jobs(job_ids)
		else:
			time.sleep(retry_interval)

		job_ids, _, retry_ids = retry_jobs_if_failed(job_ids, sh_filenames, output_paths)
		all_job_ids.extend(retry_ids)
		if len(retry_ids) == 0:
			# The above is to save a call to num_jobs_in_progress (which calls check_jobs_status, which can be slow) and guard against any retried jobs that fail quickly and so are no longer in progress but need to be tried again
			jobs_in_progress = num_jobs_in_progress(job_ids)
			if jobs_in_progress == 0:
				done = True
			else:
				print("{}: Waiting for {} jobs...".format(datetime.datetime.now(), jobs_in_progress))

	return all_job_ids


def clean_up(job_ids=None, sh_filenames=None):
	if job_ids is not None:
		for job_id in job_ids:
			os.remove("slurm-{}.out".format(job_id))
	if sh_filenames is not None:
		for filename in sh_filenames:
			os.remove(filename)


def move_job_to_test(job_id):
	scontrol_out = subprocess.run(["scontrol", "show", "job", str(job_id)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.splitlines()
	sh_filename = [line for line in scontrol_out if "Command=" in line][0].split("=")[-1]
	with open(sh_filename, "r") as f:
		sh_contents = [line.strip() for line in f]
	sh_contents = [line for line in sh_contents if "--time=" not in line]
	sh_contents.insert(2, "#SBATCH --time=4:0:0")
	sh_contents.insert(2, "#SBATCH -p test")
	with open(sh_filename, "w") as f:
		for line in sh_contents:
			f.write(line + "\n")
	new_job_id = submit_sh(sh_filename)
	subprocess.run(["scancel", str(job_id)])
	return new_job_id


def move_jobs_to_test(job_ids):
	for job_id in job_ids:
		new_id = move_job_to_test(job_id)
		wait_for_jobs([new_id])
