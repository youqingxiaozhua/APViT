import os
import os.path as osp

from mmcv.runner import master_only, HOOKS, LoggerHook


@HOOKS.register_module()
class VisualDLLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        afs_path = os.getenv("VDL_LOG_PATH")
        if afs_path:
            log_dir = afs_path.replace("afs://PUBLIC_KM_WD_Data:PUBLIC_km_wd_2020@wudang.afs.baidu.com:9902/user/PUBLIC_KM_WD_Data", 'afs')
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        try:
            from visualdl import LogWriter
        except ImportError:
            raise ImportError('Please install visualdl to use '
                                'VisualDLLoggerHook.')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'VisualDl_logs')
        else:
            self.log_dir = osp.join(self.log_dir, runner.work_dir.split('/')[-1])
        print('VisualDL log dir: ', self.log_dir)
        self.writer = LogWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        self.writer.close()
