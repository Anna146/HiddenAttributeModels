# smaller version of sacred's SqlObserver that logs fewer details to the DB
from __future__ import division, print_function, unicode_literals
import hashlib
import json
import os
import threading

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sacred.commandline_options import CommandLineOption
from sacred.dependencies import get_digest
from sacred.observers.base import RunObserver

DEFAULT_SQL_PRIORITY = 100

# ################################ ORM ###################################### #
Base = declarative_base()


class Source(Base):
    __tablename__ = 'source'

    @classmethod
    def get_or_create(cls, filename, md5sum, session):
        instance = session.query(cls).filter_by(filename=filename,
                                                md5sum=md5sum).first()
        if instance:
            return instance
        md5sum_ = get_digest(filename)
        assert md5sum_ == md5sum, 'Weird: found md5 mismatch for {}: {} != {}'\
            .format(filename, md5sum, md5sum_)
        with open(filename, 'r') as f:
            return cls(filename=filename, md5sum=md5sum, content=f.read())

    source_id = sa.Column(sa.Integer, primary_key=True)
    filename = sa.Column(sa.String(256))
    md5sum = sa.Column(sa.String(32))
    content = sa.Column(sa.Text)

    def to_json(self):
        return {'filename': self.filename,
                'md5sum': self.md5sum}


class Dependency(Base):
    __tablename__ = 'dependency'

    @classmethod
    def get_or_create(cls, dep, session):
        name, _, version = dep.partition('==')
        instance = session.query(cls).filter_by(name=name,
                                                version=version).first()
        if instance:
            return instance
        return cls(name=name, version=version)

    dependency_id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String(32))
    version = sa.Column(sa.String(16))

    def to_json(self):
        return "{}=={}".format(self.name, self.version)


class Artifact(Base):
    __tablename__ = 'artifact'

    @classmethod
    def create(cls, name, filename):
        with open(filename, 'rb') as f:
            return cls(filename=name, content=f.read())

    artifact_id = sa.Column(sa.Integer, primary_key=True)
    filename = sa.Column(sa.String(64))
    content = sa.Column(sa.LargeBinary)

    run_id = sa.Column(sa.Integer, sa.ForeignKey('run.run_id'))
    run = sa.orm.relationship("Run", backref=sa.orm.backref('artifacts'))

    def to_json(self):
        return {'_id': self.artifact_id,
                'filename': self.filename}


class Resource(Base):
    __tablename__ = 'resource'

    @classmethod
    def get_or_create(cls, filename, session):
        md5sum = get_digest(filename)
        instance = session.query(cls).filter_by(filename=filename,
                                                md5sum=md5sum).first()
        if instance:
            return instance
        with open(filename, 'rb') as f:
            return cls(filename=filename, md5sum=md5sum, content=f.read())

    resource_id = sa.Column(sa.Integer, primary_key=True)
    filename = sa.Column(sa.String(256))
    md5sum = sa.Column(sa.String(32))
    content = sa.Column(sa.LargeBinary)

    def to_json(self):
        return {'filename': self.filename,
                'md5sum': self.md5sum}


class Host(Base):
    __tablename__ = 'host'

    @classmethod
    def get_or_create(cls, host_info, session):
        h = dict(
            hostname=host_info['hostname'],
            cpu=host_info['cpu'],
            os=host_info['os'][0],
            os_info=host_info['os'][1],
            python_version=host_info['python_version']
        )

        return session.query(cls).filter_by(**h).first() or cls(**h)

    host_id = sa.Column(sa.Integer, primary_key=True)
    cpu = sa.Column(sa.String(64))
    hostname = sa.Column(sa.String(64))
    os = sa.Column(sa.String(16))
    os_info = sa.Column(sa.String(64))
    python_version = sa.Column(sa.String(16))

    def to_json(self):
        return {'cpu': self.cpu,
                'hostname': self.hostname,
                'os': [self.os, self.os_info],
                'python_version': self.python_version}


experiment_source_association = sa.Table(
    'experiments_sources', Base.metadata,
    sa.Column('experiment_id', sa.Integer,
              sa.ForeignKey('experiment.experiment_id')),
    sa.Column('source_id', sa.Integer, sa.ForeignKey('source.source_id'))
)

experiment_dependency_association = sa.Table(
    'experiments_dependencies', Base.metadata,
    sa.Column('experiment_id', sa.Integer,
              sa.ForeignKey('experiment.experiment_id')),
    sa.Column('dependency_id', sa.Integer,
              sa.ForeignKey('dependency.dependency_id'))
)


class Experiment(Base):
    __tablename__ = 'experiment'

    @classmethod
    def get_or_create(cls, ex_info, session):
        name = ex_info['name']
        # Compute a MD5sum of the ex_info to determine its uniqueness
        h = hashlib.md5()
        h.update(json.dumps(ex_info).encode())
        md5 = h.hexdigest()
        instance = session.query(cls).filter_by(name=name, md5sum=md5).first()
        if instance:
            return instance

        #dependencies = [Dependency.get_or_create(d, session)
        #                for d in ex_info['dependencies']]
        #sources = [Source.get_or_create(s, md5sum, session)
        #           for s, md5sum in ex_info['sources']]
        dependencies = []
        sources = []

        return cls(name=name, dependencies=dependencies, sources=sources,
                   md5sum=md5, base_dir=ex_info['base_dir'])

    experiment_id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String(32))
    md5sum = sa.Column(sa.String(32))
    base_dir = sa.Column(sa.String(64))
    sources = sa.orm.relationship("Source",
                                  secondary=experiment_source_association,
                                  backref="experiments")
    dependencies = sa.orm.relationship(
        "Dependency",
        secondary=experiment_dependency_association,
        backref="experiments")

    def to_json(self):
        return {'name': self.name,
                'base_dir': self.base_dir,
                'sources': [s.to_json() for s in self.sources],
                'dependencies': [d.to_json() for d in self.dependencies]}


run_resource_association = sa.Table(
    'runs_resources', Base.metadata,
    sa.Column('run_id', sa.Integer, sa.ForeignKey('run.run_id')),
    sa.Column('resource_id', sa.Integer, sa.ForeignKey('resource.resource_id'))
)


class Metric(Base):
    __tablename__ = 'metric'
    metric_id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String(64))
    step = sa.Column(sa.Integer)
    value = sa.Column(sa.Float)
    timestamp = sa.Column(sa.DateTime)

    run_id = sa.Column(sa.Integer, sa.ForeignKey('run.run_id', ondelete="CASCADE"))
    run = sa.orm.relationship("Run")
    
    
class Run(Base):
    __tablename__ = 'run'

    run_id = sa.Column(sa.Integer, primary_key=True)

    command = sa.Column(sa.String(64))

    # times
    start_time = sa.Column(sa.DateTime)
    heartbeat = sa.Column(sa.DateTime)
    stop_time = sa.Column(sa.DateTime)
    queue_time = sa.Column(sa.DateTime)

    # meta info
    priority = sa.Column(sa.Float)
    comment = sa.Column(sa.Text)

    fail_trace = sa.Column(sa.Text)

    # Captured out
    captured_out = sa.Column(sa.Text)

    # Configuration & info
    config = sa.Column(sa.JSON)
    info = sa.Column(sa.JSON)

    status = sa.Column(sa.Enum("RUNNING", "COMPLETED", "INTERRUPTED",
                               "TIMEOUT", "FAILED", "QUEUED", "MANUAL", name='statuses'))
    tries = sa.Column(sa. Integer, default=0)

    host_id = sa.Column(sa.Integer, sa.ForeignKey('host.host_id'))
    host = sa.orm.relationship("Host", backref=sa.orm.backref('runs'))

    experiment_id = sa.Column(sa.Integer,
                              sa.ForeignKey('experiment.experiment_id'))
    experiment = sa.orm.relationship("Experiment",
                                     backref=sa.orm.backref('runs'))

    # artifacts = backref
    resources = sa.orm.relationship("Resource",
                                    secondary=run_resource_association,
                                    backref="runs")

    result = sa.Column(sa.Float)

    def to_json(self):
        return {
            '_id': self.run_id,
            'command': self.command,
            'start_time': self.start_time,
            'heartbeat': self.heartbeat,
            'stop_time': self.stop_time,
            'queue_time': self.queue_time,
            'status': self.status,
            'result': self.result,
            'meta': {
                'comment': self.comment,
                'priority': self.priority},
            #'resources': [r.to_json() for r in self.resources],
            #'artifacts': [a.to_json() for a in self.artifacts],
            #'host': self.host.to_json(),
            'experiment': self.experiment.to_json(),
            'config': self.config,
            'captured_out': self.captured_out,
            'fail_trace': self.fail_trace,
        }


# ############################# Observer #################################### #

class CustomSqlObserver(RunObserver):
    @classmethod
    def create(cls, url, echo=False, priority=DEFAULT_SQL_PRIORITY):
        engine = sa.create_engine(url, echo=echo)#, pool_pre_ping=True)
        return cls(engine, sessionmaker(bind=engine)(), priority)

    def __init__(self, engine, session, priority=DEFAULT_SQL_PRIORITY):
        self.engine = engine
        self.session = session
        self.priority = priority
        self.run = None
        self.lock = threading.Lock()

    def started_event(self, ex_info, command, host_info, start_time, config,
                      meta_info, _id):
        Base.metadata.create_all(self.engine)
        sql_exp = Experiment.get_or_create(ex_info, self.session)
        sql_host = Host.get_or_create(host_info, self.session)

        if 'overwrite' in meta_info:
            r = meta_info['overwrite']
            # make sure run isn't marked as completed
            if r.status == "COMPLETED":
                raise RuntimeError("tried to overwrite completed run: %s" % r.to_json())

            self.run = r
            self.run.status = 'RUNNING'
            self.run.start_time = start_time
            self.run.config = config
            self.run.command = command
            self.run.priority = meta_info.get('priority', 0)
            self.run.comment = meta_info.get('comment', '')
            self.run.experiment = sql_exp
            self.run.host = sql_host
        else:
            if _id is None:
                # we use an autoincrement run_id, which means we don't accept an id from other observers
                pass
                #i = self.session.query(Run).order_by(Run.run_id.desc()).with_for_update().first()
                #_id = 0 if i is None else i.run_id + 1
            else:
                raise RuntimeError("this observer does not support using an existing _id")

            self.run = Run(#run_id=_id,
                           start_time=start_time,
                           config=config,
                           command=command,
                           priority=meta_info.get('priority', 0),
                           comment=meta_info.get('comment', ''),
                           experiment=sql_exp,
                           host=sql_host,
                           status='RUNNING')

        self.run.tries = 1 if self.run.tries is None else self.run.tries + 1
        self.save()
        #return _id or self.run.run_id
        _id = self.run.run_id
        return _id

    def queued_event(self, ex_info, command, host_info, queue_time, config,
                     meta_info, _id):

        Base.metadata.create_all(self.engine)
        sql_exp = Experiment.get_or_create(ex_info, self.session)
        sql_host = Host.get_or_create(host_info, self.session)
        
        if _id is None:
            # we use an autoincrement run_id, which means we don't accept an id from other observers
            pass
            #i = self.session.query(Run).order_by(Run.run_id.desc()).with_for_update().first()
            #_id = 0 if i is None else i.run_id + 1
        else:
            raise RuntimeError("this observer does not support using an existing _id")

        self.run = Run(#run_id=_id,
                       config=config,
                       command=command,
                       priority=meta_info.get('priority', 0),
                       comment=meta_info.get('comment', ''),
                       experiment=sql_exp,
                       host = sql_host,
                       status='QUEUED')
        self.save()
        #return _id or self.run.run_id
        _id = self.run.run_id
        return _id

    def log_metrics(self, metrics_by_name, info):
        for name, d in metrics_by_name.items():
            # we manually lock and commit, because the locking in save() happens once for every added object
            with self.lock:
                for step, val, timestamp in zip(d['steps'], d['values'], d['timestamps']):
                    m = Metric(run_id=self.run.run_id, name=name,
                            step=step, value=val, timestamp=timestamp)
                    self.session.add(m)
                    #self.save(add=m)
                self.session.commit()

    def heartbeat_event(self, info, captured_out, beat_time, result):
        if self.run is None:
            print('WARNING: skipping heartbeat because self.run is not yet initialized')
            return
        self.run.info = info
        #self.run.captured_out = captured_out
        self.run.heartbeat = beat_time
        self.run.result = result
        #print(threading.current_thread(), "HB save")
        self.save()

    def completed_event(self, stop_time, result):
        self.run.stop_time = stop_time
        self.run.result = result
        self.run.status = 'COMPLETED'
        print(threading.current_thread(), "SO complete")
        self.save()

    def interrupted_event(self, interrupt_time, status):
        self.run.stop_time = interrupt_time
        self.run.status = status
        print(threading.current_thread(), "SO interrupt")
        self.save()

    def failed_event(self, fail_time, fail_trace):
        self.run.stop_time = fail_time
        self.run.fail_trace = '\n'.join(fail_trace)
        self.run.status = 'FAILED'
        print(threading.current_thread(), "SO fail")
        self.save()

    def resource_event(self, filename):
        return
        #res = Resource.get_or_create(filename, self.session)
        #self.run.resources.append(res)
        #self.save()

    def artifact_event(self, name, filename):
        return
        #a = Artifact.create(name, filename)
        #self.run.artifacts.append(a)
        #self.save()

    def save(self, add=None):
        with self.lock:
            self.session.add(self.run)
            if add is not None:
                self.session.add(add)
            self.session.commit()

    def query(self, _id):
        run = self.session.query(Run).filter_by(id=_id).first()
        return run.to_json()

    def __eq__(self, other):
        if isinstance(other, SqlObserver):
            # fixme: this will probably fail to detect two equivalent engines
            return (self.engine == other.engine and
                    self.session == other.session)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


# ######################## Commandline Option ############################### #

class SqlOption(CommandLineOption):
    """Add a SQL Observer to the experiment."""

    arg = 'DB_URL'
    arg_description = \
        "The typical form is: dialect://username:password@host:port/database"

    @classmethod
    def apply(cls, args, run):
        run.observers.append(SqlObserver.create(args))
