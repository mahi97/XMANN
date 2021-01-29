from head.static_read_head import StaticReadHead
from head.static_write_head import StaticWriteHead
from head.dynamic_write_head import DynamicWriteHead
from head.dynamic_read_head import DynamicReadHead

HEADS = {
    'static-read': StaticReadHead,
    'static-write': StaticWriteHead,
    'dynamic-read': DynamicReadHead,
    'dynamic-write': DynamicWriteHead
}