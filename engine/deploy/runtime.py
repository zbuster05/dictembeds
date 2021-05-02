import sys
import argparse
from gunicorn.app.base import BaseApplication

class Application(BaseApplication):
    def load_config(self, port=18860, numWorkers=1):
        s = self.cfg.set
        s('bind', "0.0.0.0:"+str(port))
        s('workers', numWorkers)
        s('keepalive', 60)
        s('timeout', 6000)
        s('accesslog', "-")
        s('access_log_format', '%(t)s %(h)s "%(r)s" %(s)s %(b)s %(D)s "%(a)s"')

    def load(self):
        from Engine import app
        return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="InscriptioEngine", description='The ML Engine that Powers Inscriptio by Hugging its Face')
    parser.add_argument('command', type=str, help='the command to execute (server|summarize)')
    parser.add_argument('--port', '-p', type=int, default=18860, help='the port to run the server (default 18860)')
    parser.add_argument('--workers', '-w', type=int, default=4, help='the number of webserver workers (default 4, not Engine workers!)')
    parser.add_argument('--passages', '-i', type=str, nargs='+', help='the input passages')
    parser.add_argument('--titles', '-t', type=str, nargs='+', help='the input titles')
    parser.add_argument('--timestats', dest='timestats', action='store_true', help="print time statistics in summary mode (default off)")
    parser.set_defaults(timestats=False)


    args = parser.parse_args()

    if (args.command == "server"):
        Application(args.port, args.workers).run()
    elif (args.command == "summarize"):
        from Engine import Engine
        import time
        t1 = time.time()
        e = Engine()
        t2 = time.time()
        res = e.batch_execute(zip(args.titles, args.passages))
        t3 = time.time()
        for sent in res:
            print(sent)
        if args.timestats:
            print(t2-t1, t3-t2, t3-t1)
    else:
        parser.print_help(sys.stderr)

