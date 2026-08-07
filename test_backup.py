from knowledge.backup import create_backup, list_backups

result = create_backup()
print("create_backup() returned:", result)

print("\nCurrent backups:")
for b in list_backups():
    print(" -", b)