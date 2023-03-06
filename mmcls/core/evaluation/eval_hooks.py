import os.path as osp
from math import inf

from mmcv.runner import Hook
from torch.utils.data import DataLoader
from mmcls.utils import get_root_logger


class EvalHook(Hook):
    """Evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, by_epoch=True, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.by_epoch = by_epoch
        self.logger = get_root_logger()

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmcls.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmcls.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        return eval_res


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=True,
                 by_epoch=True,
                 print_best=True,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        super().__init__(dataloader, interval, by_epoch, **eval_kwargs)
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs
        self.print_best = print_best
    
    def before_run(self, runner):
        if self.print_best is not None:
            if runner.meta is None:
                warnings.warn('runner.meta is None. Creating a empty one.')
                runner.meta = dict()
            runner.meta.setdefault('hook_msgs', dict())
    
    def before_train_epoch(self, runner):
        return
        # freeze IRNet
        # frozen_blocks = 7
        model = runner.model.module.convert
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return

        if frozen_blocks > 0:
            print(f'IRSE freeze the first {frozen_blocks} blocks, it has {len(model.body)} blocks ')
            model.input_layer.eval()
            print('in freeze', model.input_layer[1].training)
            for param in model.input_layer.parameters():
                param.requires_grad = False
        
        for i in range(frozen_blocks):
            m = model.body[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmcls.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            eval_res = self.evaluate(runner, results)
            if self.print_best:
                best_score = runner.meta['hook_msgs'].get('best_score', -inf)
                if 'top-1' not in eval_res:
                    return
                acc = eval_res['top-1']
                if acc > best_score:
                    self.logger.info(f'top-1 accuracy improved from {best_score} to {acc}')
                    runner.meta['hook_msgs']['best_score'] = acc
                    runner.save_checkpoint(runner.work_dir, save_optimizer=False, filename_tmpl='best.pth', create_symlink=False)
                else:
                    self.logger.info(f'top-1 accuracy did not improve from {best_score}')

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmcls.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            eval_res = self.evaluate(runner, results)
            if self.print_best:
                best_score = runner.meta['hook_msgs'].get('best_score', -inf)
                if 'top-1' not in eval_res:
                    return
                acc = eval_res['top-1']
                if acc > best_score:
                    self.logger.info(f'top-1 accuracy improved from {best_score} to {acc}')
                    runner.meta['hook_msgs']['best_score'] = acc
                    runner.save_checkpoint(runner.work_dir, save_optimizer=False, filename_tmpl='best.pth', create_symlink=False)
                else:
                    self.logger.info(f'top-1 accuracy did not improve from {best_score}')
