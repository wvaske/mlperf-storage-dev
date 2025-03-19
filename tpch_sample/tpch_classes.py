
class RunTPCH(LoggedObject):

    # tpc-h spec appendix a
    APPENDIX_A = {'all_queries': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                  'power_test': [14, 2, 9, 20, 6, 17, 18, 8, 21, 13, 3, 22, 16, 4, 11, 15, 1, 10, 19, 5, 7, 12],
                  'throughput_test': [[21, 3, 18, 5, 11, 7, 6, 20, 17, 12, 16, 15, 13, 10, 2, 8, 14, 19, 9, 22, 1, 4],
                                      [6, 17, 14, 16, 19, 10, 9, 2, 15, 8, 5, 22, 12, 7, 13, 18, 1, 4, 20, 3, 11, 21],
                                      [8, 5, 4, 6, 17, 7, 1, 18, 22, 14, 9, 10, 15, 11, 20, 2, 21, 19, 13, 16, 12, 3],
                                      [5, 21, 14, 19, 15, 17, 12, 6, 4, 9, 8, 16, 11, 2, 10, 18, 1, 13, 7, 22, 3, 20],
                                      [21, 15, 4, 6, 7, 16, 19, 18, 14, 22, 11, 13, 3, 1, 2, 5, 8, 20, 12, 17, 10, 9],
                                      [10, 3, 15, 13, 6, 8, 9, 7, 4, 11, 22, 18, 12, 1, 5, 16, 2, 14, 19, 20, 17, 21],
                                      [18, 8, 20, 21, 2, 4, 22, 17, 1, 11, 9, 19, 3, 13, 5, 7, 10, 16, 6, 14, 15, 12],
                                      [19, 1, 15, 17, 5, 8, 9, 12, 14, 7, 4, 3, 20, 16, 6, 22, 10, 13, 2, 21, 18, 11],
                                      [8, 13, 2, 20, 17, 3, 6, 21, 18, 11, 19, 10, 15, 4, 22, 1, 7, 12, 9, 14, 5, 16],
                                      [6, 15, 18, 17, 12, 1, 7, 2, 22, 13, 21, 10, 14, 9, 3, 16, 20, 19, 11, 4, 8, 5],
                                      [15, 14, 18, 17, 10, 20, 16, 11, 1, 8, 4, 22, 5, 12, 3, 9, 21, 2, 13, 6, 19, 7],
                                      [1, 7, 16, 17, 18, 22, 12, 6, 8, 9, 11, 4, 2, 5, 20, 21, 13, 10, 19, 3, 14, 15],
                                      [21, 17, 7, 3, 1, 10, 12, 22, 9, 16, 6, 11, 2, 4, 5, 14, 8, 20, 13, 18, 15, 19],
                                      [2, 9, 5, 4, 18, 1, 20, 15, 16, 17, 7, 21, 13, 14, 19, 8, 22, 11, 10, 3, 12, 6],
                                      [16, 9, 17, 8, 14, 11, 10, 12, 6, 21, 7, 3, 15, 5, 22, 20, 1, 13, 19, 2, 4, 18],
                                      [1, 3, 6, 5, 2, 16, 14, 22, 17, 20, 4, 9, 10, 11, 15, 8, 12, 19, 18, 13, 7, 21],
                                      [3, 16, 5, 11, 21, 9, 2, 15, 10, 18, 17, 7, 8, 19, 14, 13, 1, 4, 22, 20, 6, 12],
                                      [14, 4, 13, 5, 21, 11, 8, 6, 3, 17, 2, 20, 1, 19, 10, 9, 12, 18, 15, 7, 22, 16],
                                      [4, 12, 22, 14, 5, 15, 16, 2, 8, 10, 17, 9, 21, 7, 3, 6, 13, 18, 11, 20, 19, 1],
                                      [16, 15, 14, 13, 4, 22, 18, 19, 7, 1, 12, 17, 5, 10, 20, 3, 9, 21, 11, 2, 6, 8],
                                      [20, 14, 21, 12, 15, 17, 4, 19, 13, 10, 11, 1, 16, 5, 18, 7, 8, 22, 9, 6, 3, 2],
                                      [16, 14, 13, 2, 21, 10, 11, 4, 1, 22, 18, 12, 19, 5, 7, 8, 6, 3, 15, 20, 9, 17],
                                      [18, 15, 9, 14, 12, 2, 8, 11, 22, 21, 16, 1, 6, 17, 5, 10, 19, 4, 20, 13, 3, 7],
                                      [7, 3, 10, 14, 13, 21, 18, 6, 20, 4, 9, 8, 22, 15, 2, 1, 5, 12, 19, 17, 11, 16],
                                      [18, 1, 13, 7, 16, 10, 14, 2, 19, 5, 21, 11, 22, 15, 8, 17, 20, 3, 4, 12, 6, 9],
                                      [13, 2, 22, 5, 11, 21, 20, 14, 7, 10, 4, 9, 19, 18, 6, 3, 1, 8, 15, 12, 17, 16],
                                      [14, 17, 21, 8, 2, 9, 6, 4, 5, 13, 22, 7, 15, 3, 1, 18, 16, 11, 10, 12, 20, 19],
                                      [10, 22, 1, 12, 13, 18, 21, 20, 2, 14, 16, 7, 15, 3, 4, 17, 5, 19, 6, 8, 9, 11],
                                      [10, 8, 9, 18, 12, 6, 1, 5, 20, 11, 17, 22, 16, 3, 13, 2, 15, 21, 14, 19, 7, 4],
                                      [7, 17, 22, 5, 3, 10, 13, 18, 9, 1, 14, 15, 21, 19, 16, 12, 8, 6, 11, 20, 4, 2],
                                      [2, 9, 21, 3, 4, 7, 1, 11, 16, 5, 20, 19, 18, 8, 17, 13, 10, 12, 15, 6, 14, 22],
                                      [15, 12, 8, 4, 22, 13, 16, 17, 18, 3, 7, 5, 6, 1, 9, 11, 21, 10, 14, 20, 19, 2],
                                      [15, 16, 2, 11, 17, 7, 5, 14, 20, 4, 21, 3, 10, 9, 12, 8, 13, 6, 18, 19, 22, 1],
                                      [1, 13, 11, 3, 4, 21, 6, 14, 15, 22, 18, 9, 7, 5, 10, 20, 12, 16, 17, 8, 19, 2],
                                      [14, 17, 22, 20, 8, 16, 5, 10, 1, 13, 2, 21, 12, 9, 4, 18, 3, 7, 6, 19, 15, 11],
                                      [9, 17, 7, 4, 5, 13, 21, 18, 11, 3, 22, 1, 6, 16, 20, 14, 15, 10, 8, 2, 12, 19],
                                      [13, 14, 5, 22, 19, 11, 9, 6, 18, 15, 8, 10, 7, 4, 17, 16, 3, 1, 12, 2, 21, 20],
                                      [20, 5, 4, 14, 11, 1, 6, 16, 8, 22, 7, 3, 2, 12, 21, 19, 17, 13, 10, 15, 18, 9],
                                      [3, 7, 14, 15, 6, 5, 21, 20, 18, 10, 4, 16, 19, 1, 13, 9, 8, 17, 11, 12, 22, 2],
                                      [13, 15, 17, 1, 22, 11, 3, 4, 7, 20, 14, 21, 9, 8, 2, 18, 16, 6, 10, 12, 5, 19]]
                  }

    def __init__(self, auth, db_connection_parameters, dop, run_uuid, data_location, query_location, qgen_location,
                 scale_factor, queries=None, spec_run=True, randomize=False, streams=1, seed=None, *args, **kwargs):
        """
        This executes the full TPCH test

        :param auth: authentication object
        :param db_connection_parameters: dictionary of values to create DB controls
        :param dop: degrees of parallelism
        :param run_uuid: run_uuid
        :param data_location: location to store test execution output
        :param query_location: location of preset query files
        :param qgen_location: location of qgen binary
        :param scale_factor: scale factor of the dataset
        :param queries: list of queries to run in the TPCH sequence
        :param spec_run: True/False, If True, randomize is ignored and queries are executed in the order specified for
                        for the corresponding stream number.
        :param randomize: True/False. Randomize list of queries provided for each stream. Parameter is ignored if
                        sepc_run is True
        :param streams: number of streams
        :param seed: seed value to initialize qgen paramters generation
        :param args:
        :param kwargs:
        """

        super().__init__(*args, **kwargs)
        self.auth = auth
        self.db_connection_parameters = db_connection_parameters

        if not queries or queries == [None]:
            self.queries = self.APPENDIX_A['all_queries']
        else:
            self.queries = queries

        self.streams = streams
        self.seed = seed
        self.run_uuid = run_uuid
        self.dop = dop
        self.scale_factor = scale_factor
        self.randomize = randomize
        self.spec_run = spec_run
        self.data_location = data_location
        self.query_location = query_location
        self.qgen_location = qgen_location
        self.log_queue = multiprocessing.Manager().Queue()
        self.log_queue_listener = logging.handlers.QueueListener(
            self.log_queue, *self.logger.handlers, respect_handler_level=True
        )

    @staticmethod
    def _create_run_stream(kwargs):
        db_connection_parameters = kwargs.get('db_connection_parameters')

        # creating log queue to enable logging for each execution thread
        if not kwargs.get('log_queue'):
            raise Exception(f'Log not provided for tpch_runner')
        logger = get_queued_logger(kwargs['log_queue'])

        # Initialize TPCHStreamRunner
        db_driver = get_db_driver(db_connection_parameters.get('db_system'),
                                                      auth=kwargs.get('auth'),
                                                      logger=logger,
                                                      **db_connection_parameters)
        tpch_runner = TPCHStreamRunner(db_ctl=db_driver, **kwargs)

        tpch_runner.db_ctl.connect()
        tpch_runner.logger.status(f'Stream {tpch_runner.stream_id} executing Queries:{pformat(tpch_runner.queries)}')

        # streamdata is a list of tuples containing information about stream and query execution times
        streamdata = tpch_runner.run_stream()

        # time is extracted here for logging purposes
        times = [x for _, _, _, x in streamdata]  # x is the execution time to run a query
        tpch_runner.logger.status(f'Stream {tpch_runner.stream_id} total runtime {str_time(sum(times))}')
        tpch_runner.logger.verboser(f'Stream {tpch_runner.stream_id} Runtimes: {pformat(streamdata)}')
        tpch_runner.db_ctl.disconnect()

        return streamdata

    def execute(self):
        """Executes the TPCH workload"""

        def _append_result(result):
            """
            Callback method for map_async

            """
            for r in result:
                if r is not None:
                    all_stream_data.append(r)

        all_stream_data = []
        stream_kwargs = []

        # generating seed value based on tpc-h 2.1.3.3
        self.seed = int(datetime.now().strftime('%m%d%H%M%S')) if not self.seed else self.seed

        for stream in range(self.streams):
            if self.spec_run and self.streams == 1:
                queries = self.APPENDIX_A['power_test']
                self.logger.debug("Selecting 'power_test' queries in RunTPCH execute method")
            elif self.spec_run and self.streams > 1:
                queries = self.APPENDIX_A['throughput_test'][stream]
                self.logger.debug("Selecting 'throughput_test' queries in RunTPCH execute method")
                self.seed += stream + 1
            else:
                queries = copy.deepcopy(self.queries)
                self.logger.debug('Doing deep copy of queries in RunTPCH execute mdethod')
                self.seed += stream + 1
                if self.randomize and len(queries) > 1:
                    self.logger.ridiculous(f'Shuffling stream {stream + 1} queries with seed {self.seed}')
                    random.seed(self.seed)
                    random.shuffle(queries)

            # intializing stream parameters
            stream_kwargs.append({'db_connection_parameters': self.db_connection_parameters,
                                  'queries': queries,
                                  'seed': self.seed,
                                  'stream_id': stream + 1,
                                  'log_queue': self.log_queue,
                                  'auth': self.auth,
                                  'query_location': self.query_location,
                                  'qgen_location': self.qgen_location,
                                  'scale_factor': self.scale_factor,
                                  'dop': self.dop})

        self.log_queue_listener.start()

        pool = multiprocessing.Pool(processes=self.streams)
        results = pool.map_async(self._create_run_stream, stream_kwargs, callback=_append_result)
        results.get()
        pool.close()
        pool.join()

        self.log_queue_listener.stop()

        for s in all_stream_data:
            if isinstance(s, Exception):
                return s

        result_dict = dict()

        stream_times = []
        self.logger.ridiculous(f'All data: {pformat(all_stream_data)}')
        for s_data in all_stream_data:
            self.logger.plaid(f'Stream data: {pformat(s_data)}')
            s_time = 0
            # extract data from results an put into list of dictionaries
            for s_id, s_seed, q_num, q_time in s_data:
                if not result_dict.get(f'stream_{s_id}'):
                    result_dict[f'stream_{s_id}'] = dict()
                    result_dict[f'stream_{s_id}']['seed'] = s_seed
                result_dict[f'stream_{s_id}'][f'query_{q_num}'] = q_time
                s_time += q_time
                result_dict[f'stream_{s_id}']['stream_time'] = s_time
            stream_times.append(s_time)

        result_dict['avg_stream_time'] = numpy.mean(stream_times)
        result_dict['max_stream_time'] = numpy.max(stream_times)
        result_dict['std_stream_time'] = numpy.std(stream_times) if self.streams > 1 else 0
        result_dict['scale_factor'] = self.scale_factor
        result_dict['dop'] = self.dop
        if len(self.queries) == 22:
            result_dict['QPH'] = (self.streams * 22) / (result_dict['avg_stream_time'] / 3600)

        data_file = os.path.join(self.data_location, f'{self.run_uuid}.tpch.json')
        with open(data_file, 'w') as f:
            json.dump(result_dict, f, sort_keys=True, indent=2)

        result = {'filename': data_file,
                  'auth': LOCAL_AUTH,
                  'aggregation_group': 'tpch.json'}

        return result


class TPCHStreamRunner(LoggedObject):
    def __init__(self, db_ctl, stream_id, query_location, qgen_location, queries, scale_factor, dop, seed,
                 log_queue=None, *args, **kwargs):
        """
        This executes implencts the execution of a single TPCH stream

        :param db_ctl: DBControl to execute queries
        :param stream_id: stream_id
        :param query_location: full filepath of preset queries
        :param qgen_location: full file path of qgen binary
        :param queries: queries to execute
        :param scale_factor: scale factor of dataset
        :param dop: degrees of parallelism
        :param seed: value to seed random vales for qgen query population
        :param log_queue: log queue
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.db_ctl = db_ctl
        self.dop = dop
        self.scale_factor = scale_factor
        self.query_location = query_location
        self.qgen_location = qgen_location
        self.queries = queries
        self.seed = seed
        self.stream_id = stream_id
        self.logger = get_queued_logger(log_queue)

        self.logger.ludicrous(f'Queries in TPCHStreamRunner: {self.queries}')

    def gen_query(self, query_number):
        """
        Populates a given query with data generated by qgen

        :param query_number: query number
        :return:
        """

        command = f'cd {self.qgen_location} && DSS_QUERY={self.query_location} ./qgen -s {self.scale_factor} {query_number} -r {self.seed}'
        self.logger.plaid(f'Generating query with command: {command}')
        cmd = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
        out, err = cmd.communicate()
        if err:
            self.logger.error(f'QGEN execution failed generating query {query_number} '
                              f'with stream_id {self.stream_id}:\n{err}')
            return Exception(f'QGEN execution failed generating query {query_number} '
                             f'with stream_id {self.stream_id}:\n{err}')
        else:
            return out

    def run_query(self, query_number):
        """
        Executes a specified query

        :param query_number:
        :return:
        """
        self.logger.status(f'Stream {self.stream_id} Running Query {query_number}')

        # each query has special tags that need to be replaced
        # sse_xyz is used to input parallel value
        # @123 places a unique value for views
        raw_query = self.gen_query(query_number).decode()
        query = raw_query.replace('sse_xyz', str(self.dop)).replace('@123', str(self.stream_id))

        start = time.time()
        self.logger.ridiculous(f'Stream {self.stream_id} Query {query_number} Start Time: {start}')

        # this is a workaround for query 15. Query 15 is actually 3 separate queries
        for q in query.split(';'):
            try:
                self.db_ctl.execute(q)
            except Exception as e:
                self.logger.error(f'Exception from stream {self.stream_id} in running query number {query_number}: {e}')
                return Exception(f'Exception from stream {self.stream_id} in running query number {query_number}: {e}')

        end = time.time()
        self.logger.ridiculous(f'Stream {self.stream_id} Query {query_number} End Time: {end}')
        run_time = end - start
        self.logger.verbosest(f'Stream {self.stream_id} Query {query_number} Run Time: {str_time(run_time)}')
        return run_time

    def run_stream(self):
        # return a list of tuples
        return [(self.stream_id, self.seed, stream_query, self.run_query(stream_query)) for stream_query in
                self.queries]
