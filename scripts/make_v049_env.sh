bash -lc '
set -eu
cd /home/yoichi/develop/ai/columnar_ed_ann

TS="$(date +%Y%m%d_%H%M%S)"

# 1) 現在のcolumnar_ed_ann.pyをv049としてコピー
cp columnar_ed_ann.py columnar_ed_ann_v049.py

# 2) code_backup作成 + 現在のcolumnar_ed_ann.pyを退避
mkdir -p code_backup
mv columnar_ed_ann.py "code_backup/columnar_ed_ann_backup_${TS}.py"

# 3) modules_backup作成 + 現在のmodulesをコピー退避
mkdir -p modules_backup
cp -a modules "modules_backup/modules_backup_${TS}"

# 4) 開発準備
# 4-1: 既存スクリプト互換のため、columnar_ed_ann.pyをv049への薄いラッパにしておく
cat > columnar_ed_ann.py <<'"'"'PY'"'"'
#!/usr/bin/env python3
import runpy
runpy.run_path("columnar_ed_ann_v049.py", run_name="__main__")
PY
chmod +x columnar_ed_ann.py

# 4-2: 今後のバックアップ運用ルールを明文化
cat > BACKUP_POLICY.md <<EOF
# Backup Policy

## code backup
- Main script backup location: \`code_backup/\`
- Naming: \`columnar_ed_ann_backup_YYYYMMDD_HHMMSS.py\`

## modules backup
- Modules backup location: \`modules_backup/\`
- Naming: \`modules_backup_YYYYMMDD_HHMMSS\`

## current development target
- Primary development file: \`columnar_ed_ann_v049.py\`
EOF

echo "DONE"
echo "Created: columnar_ed_ann_v049.py"
echo "Moved : code_backup/columnar_ed_ann_backup_${TS}.py"
echo "Copied: modules_backup/modules_backup_${TS}"
'
