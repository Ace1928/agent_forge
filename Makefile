smoke:
	./scripts/bootstrap.sh
	.venv/bin/python -m core.config --dir cfg --validate
	bin/eidctl state --migrate
	echo "hello" | bin/eidctl journal --add - --type note --tags smoke
	.venv/bin/python -c "from core import events as E, db as DB; E.append('state','smoke.event',{'ok':True}); DB.insert_metric('state','smoke.metric',42.0); DB.insert_journal('state','smoke.db','ok'); print('ok')"
	bin/eidosd --state-dir state --once
	bin/eidctl state --json | jq -e '.totals.note >= 1 and .files.bus >= 1'
	.venv/bin/python -m pytest -q
