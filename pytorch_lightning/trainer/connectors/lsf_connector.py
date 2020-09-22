import os
import re
import signal
from subprocess import call
from pytorch_lightning import _logger as log
from pytorch_lightning.utilities.distributed import rank_zero_info


class LSFConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(self, num_gpu_nodes):
        self.configure_lsf_ddp(num_gpu_nodes)

    def configure_lsf_ddp(self, num_gpu_nodes):
        self.trainer.is_lsf_managing_tasks = False

        # extract LSF flag vars
        # whenever we have the correct number of tasks, we let lsf manage processes
        # otherwise we launch the required number of processes
        if self.trainer.use_ddp:
            self.trainer.num_requested_gpus = self.trainer.num_gpus * num_gpu_nodes
            self.trainer.num_lsf_tasks = 0
            try:
                self.trainer.num_lsf_tasks = int(os.environ['JSM_NAMESPACE_SIZE'])
                self.trainer.is_lsf_managing_tasks = self.trainer.num_lsf_tasks == self.trainer.num_requested_gpus

                # in interactive mode we don't manage tasks
                job_name = os.environ['LSB_JOBNAME']
                if job_name == 'bash':
                    self.trainer.is_lsf_managing_tasks = False

            except Exception:
                # likely not on lsf, so set the lsf managed flag to false
                self.trainer.is_lsf_managing_tasks = False

        # used for tests only, set this flag to simulate lsf managing a task
        try:
            should_fake = int(os.environ['FAKE_LSF_MANAGING_TASKS'])
            if should_fake:
                self.trainer.is_lsf_managing_tasks = True
        except Exception:
            pass

        # notify user the that lsf is managing tasks
        if self.trainer.is_lsf_managing_tasks:
            rank_zero_info('Multi-processing is handled by Slurm.')

    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            name, numbers = root_node.split('[', maxsplit=1)
            number = numbers.split(',', maxsplit=1)[0]
            if '-' in number:
                number = number.split('-')[0]

            number = re.sub('[^0-9]', '', number)
            root_node = name + number

        return root_node

    def register_lsf_signal_handlers(self):
        # see if we're using lsf (not interactive)
        on_lsf = False
        try:
            job_name = os.environ['LSB_JOBNAME']
            if job_name != 'bash':
                on_lsf = True
        except Exception:
            pass

        if on_lsf:
            log.info('Set LSF handle signals.')
            signal.signal(signal.SIGUSR1, self.sig_handler)
            signal.signal(signal.SIGTERM, self.term_handler)

    def sig_handler(self, signum, frame):  # pragma: no-cover
        if self.trainer.is_global_zero:
            # save weights
            log.info('handling SIGUSR1')
            self.trainer.hpc_save(self.trainer.weights_save_path, self.trainer.logger)

            # find job id
            job_id = os.environ['LSB_JOBID']
            cmd = ['brequeue', job_id]

            # requeue job
            log.info(f'requeing job {job_id}...')
            result = call(cmd)

            # print result text
            if result == 0:
                log.info(f'requeued exp {job_id}')
            else:
                log.warning('requeue failed...')

            # close experiment to avoid issues
            self.trainer.logger.close()

    def term_handler(self, signum, frame):
        # save
        log.info("bypassing sigterm")
