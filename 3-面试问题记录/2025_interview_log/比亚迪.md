20250519 面试记录
1. 对最近的大模型关注是怎么样?deepseek架构介绍一下,和gpt有什么不同,GRPO算法介绍
2. deepseek与MHA跟其他的一些attention的算法上面的一些区别
3. 介绍一下你的简历里面 [xxx] 项目,用了哪些算法?怎么做的
4. [xxx] 项目里面vlm模型作业是什么?
5. 只有一个简单算法题目:
"""
将两个升序链表合并为一个新的 升序 链表并返回。
"""

```python
from typing import Optional
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeTwoLists(
    list1: Optional[ListNode], list2: Optional[ListNode]
) -> Optional[ListNode]:
    dummy = ListNode(0)
    current = dummy

    while list1 and list2:
        if list1.val < list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next

    if list1:
        current.next = list1
    else:
        current.next = list2

    return dummy.next

def solve():
    list1 = ListNode(1)
    list1.next = ListNode(2)
    list2 = ListNode(3)
    list2.next = ListNode(4)
    result = mergeTwoLists(list1, list2)
    while result:
        print(result.val, end=" -> ")
        result = result.next
    print("None")

if __name__ == "__main__":
    solve()
```

这些问题都比较简单或者私人,我就不写答案了
