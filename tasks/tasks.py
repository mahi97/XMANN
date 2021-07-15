from tasks import copy_task
from tasks import repeatcopy_task
from tasks import repeatcopy2_task
from tasks import associativerecall_task
from tasks import priority_sort_task

TASKS = {
    'copy': copy_task.CopyTask(),
    'repeat-copy': repeatcopy_task.RepeatCopyTask(),
    'repeat-copy2': repeatcopy2_task.RepeatCopyTask(),
    'associative-recall': associativerecall_task.AssociativeRecallTask(),
    'priority-sort': priority_sort_task.PrioritySortTask()
}

