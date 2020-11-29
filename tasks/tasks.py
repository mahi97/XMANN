from tasks import copy_task
from tasks import repeatcopy_task
from tasks import associativerecall_task

TASKS = {
    'copy': copy_task.CopyTask(),
    'repeat-copy': repeatcopy_task.RepeatCopyTask(),
    'associative-recall': associativerecall_task.AssociativeRecallTask()
}

