# 远端目录迁移完成

完成时间（UTC）：2026-09-27T01:04:08.712724+00:00。

实际文件根目录：/data/lyf/common/covar_kd。

| 内容 | 当前远端路径 |
|---|---|
| 当前实验项目 | /data/lyf/common/covar_kd/CoVar-KD-pair2 |
| 原始项目及共享数据 | /data/lyf/common/covar_kd/CoVar-KD |
| Python 环境 | /data/lyf/common/covar_kd/env/bin/python |
| P10 日志与检查点 | /data/lyf/common/covar_kd/CoVar-KD-pair2/runs/covar_match/P10_h20 |
| P10 报告 | /data/lyf/common/covar_kd/CoVar-KD-pair2/reports/covar_match/P10_h20 |

完整迁移 56,741 个普通文件、57,546 个目录条目，普通文件总大小 20,715,698,455 字节。复制后所有普通文件 SHA-256 与源文件一致，目录权限和软链接匹配。完整清单位于新根目录 .migration_20260927/source_manifest.json；摘要见 [migration.json](migration.json)。

迁移范围包括原根目录下全部八个顶层条目：两个项目目录、env、SETUP.md、environment-freeze.txt、upload-manifest.json、upload-verification.json、verify_upload.py。内部共享 data 和 VOCAug 链接已改为相对路径，指向同一新根目录内的原始项目。环境继续复用原有共享 Python 底座，版本和依赖未改动。

旧 /home/lyf/research_2026/covar_kd_2026 现在仅为指向新根目录的兼容软链接，供历史协议、检查点和虚拟环境脚本中的绝对路径使用。经新路径运行检查后，原文件备份已移除；文件实际存放在 /data 磁盘。

迁移时没有正式训练或 smoke 在运行；只停止了等待中的控制器。新目录冻结协议复核通过，58项CPU回归检查通过。P10 已从新目录恢复，tmux=covar-p10-h20，控制器PID=3647218，本次检查状态=WAITING_GPU。后续自动检查已切换为新路径。

原P9/P10冻结训练源码、实验矩阵、已完成结果和原补点决定均保持原样；迁移不产生新的科学结果。
