from functools import wraps
import inspect
from teradatasqlalchemy import vernumber
import re


class _QueryBand:
    """
    Class to hold the common attributes required for queryband enabling.
    """
    def __init__(self):
        self._qb_buffer = []
        self._org = "TERADATA-INTERNAL-TELEM"
        self._app_name = "TDSQLMY"
        self._app_version = vernumber.sVersionNumber
        self._qb_template = "QUERY_BAND='ORG={org};APPNAME={app_name};APPVERSION={app_version};{client_qb};'"
        self._set_qb_query_template = "SET {query_band} FOR TRANSACTION;"
        self._prev_qb_str = None
        self._prev_qb_str_freq = 0
        self._qb_regex = r'^[a-zA-Z0-9_-]+$'
        self._verbose = False

    @property
    def qb_buffer(self):
        """
        RETURNS:
            Queryband string.

        EXAMPLES:
            >>> qb = _QueryBand()
            >>> qb.qb_buffer
        """
        return self._qb_buffer

    @qb_buffer.setter
    def qb_buffer(self, query_band):
        """
        Creates query band buffer if it doesn't exist else appends query band
        to existing query band buffer self._qb_buffer.

        PARAMETERS:
            query_band

        EXAMPLES:
            >>> qb = _QueryBand()
            >>> qb.qb_buffer('ORG=TERADATA-INTERNAL-TELEM')
        """
        if not self._qb_buffer:
            self._qb_buffer = []
        self._qb_buffer.append(query_band)

    @property
    def qb_regex(self):
        """
        RETURNS:
            Regular expression which validates queryband string.

        EXAMPLES:
            >>> qb = _QueryBand()
            >>> qb.qb_regex
        """
        return self._qb_regex

    @property
    def verbose(self):
        """
        RETURNS:
            Configuration option which decides whether to print
            error logs on console or not.

        EXAMPLES:
            >>> qb = _QueryBand()
            >>> qb.verbose
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """
        Sets configuration option which decides whether to print
        error logs on console or not.

        PARAMETERS:
            verbose

        EXAMPLES:
            >>> qb = _QueryBand()
            >>> qb.verbose(True)
        """
        self._verbose = verbose

    def append_qb(self, query_band):
        """
        Creates query band buffer if it doesn't exist else appends query band
        to existing query band buffer.

        PARAMETERS:
            query_band

        EXAMPLES:
            >>> qb = _QueryBand()
            >>> qb.append_qb('ORG=TERADATA-INTERNAL-TELEM')

        """
        # When new buffer is created.
        if not self._prev_qb_str:
            self._prev_qb_str = query_band
            self._prev_qb_str_freq = 1
            return

        # check if queryband string is repeated.
        # If not, append previous queryband to buffer and update _prev_qb_str
        # and _prev_qb_str_freq else just increase the frequency of previous queryband.
        if query_band != self._prev_qb_str:
            # If _prev_qb_str is having frequency more than 1, then append queryband
            # string with frequency, else append without frequency.
            self.qb_buffer.append(self._prev_qb_str + "_" + str(self._prev_qb_str_freq)
                                  if self._prev_qb_str_freq > 1 else self._prev_qb_str)
            self._prev_qb_str = query_band
            self._prev_qb_str_freq = 1
        else:
            self._prev_qb_str_freq = self._prev_qb_str_freq + 1

    def pop_qb(self):
        """
        Removes last added queryband from query band buffer list.

        PARAMETERS:
            None

        EXAMPLES:
            >>> qb = _QueryBand()
            >>> qb.append_qb('ORG=TERADATA-INTERNAL-TELEM')
            >>> qb.pop_qb()

        """
        try:
            del self._qb_buffer[-1]
        except IndexError:
            self._qb_buffer = []

    def reset_qb(self):
        """
        Removes all querybands from query band buffer list.

        PARAMETERS:
            None

        EXAMPLES:
            >>> qb = _QueryBand()
            >>> qb.append_qb('ORG=TERADATA-INTERNAL-TELEM')
            >>> qb.reset_qb()
        """
        self._prev_qb_str = None
        self._prev_qb_str_freq = 0
        self._qb_buffer = []

    def configure_queryband_parameters(self, app_name, app_version):
        """
        DESCRIPTION:
            Configures application name and application version which
            uses queryband utility.

        PARAMETERS:
            app_name:
                Required Argument:
                Specifies name of the application which uses queryband utility.
                Types: str

            app_version:
                Required Argument:
                Specifies version of the application which uses queryband utility.
                Types: str

        RETURNS:
            None

        RAISES:
            None.

        EXAMPLES:
            >>> session_qb = _QueryBand()
            >>> session_qb.configure_queryband_parameters(app_name="TDML", app_version="20.00.00.00")
        """
        self._app_name = app_name
        self._app_version = app_version

    def generate_set_queryband_query(self):
        """
        DESCRIPTION:
            Generates a SQL query to be used while setting transaction level
            queryband for an application. Application specific data and querybands
            collected during execution of application's APIs are used while genearting
            final queryband string. Finally, cleans queryband buffer to start with
            new workflow.

        PARAMETERS:
            None

        RETURNS:
            str

        RAISES:
            None.

        EXAMPLES:
            >>> session_qb = _QueryBand()
            >>> session_qb.generate_set_queryband_query()
        """

        try:
            # Before utilizing buffer, append lazy entries
            # in _prev_qb_str to _qb_buffer.
            if self._prev_qb_str:
                # If _prev_qb_str is having frequency more than 1, then append queryband
                # string with frequency, else append without frequency.
                self.qb_buffer.append(self._prev_qb_str + "_" + str(self._prev_qb_str_freq)
                                      if self._prev_qb_str_freq > 1 else self._prev_qb_str)
            return self._set_qb_query_template.format(
                query_band=self._qb_template.format(org=self._org,
                                                    app_name=self._app_name,
                                                    app_version=self._app_version,
                                                    client_qb="APPFUNC={}".format("-".join(self._qb_buffer))))
        except Exception as append_err:
            log("Failed to generate SET QB query: ", append_err)
        finally:
            self.reset_qb()


session_queryband = _QueryBand()


def collect_queryband(queryband=None, attr=None, method=None,
                      arg_name=None, prefix=None, suffix=None):
    """
    DESCRIPTION:
        Decorator for collecting queryband string in queryband buffer.

    PARAMETERS:
        queryband:
            Optional Argument:
            Specifies queryband string.
            Types: str

        attr:
            Optional Argument:
            Specifies name of a class attribute whose value is to be used as
            queryband string.
            Types: str

        method:
            Optional Argument:
            Specifies name of a class method which returns string to be used as
            queryband string.
            Note:
                This method of class is expected to be a no-arg utility method and
                should return an expected queryband string for some processing done
                by a class/class method which needs to be tracked by queryband.
            Types: str

        arg_name:
            Optional Argument:
            Specifies name of an argument of a decorated function/method, whose value
            is to be used as queryband string.
            Types: str

        prefix:
            Optional Argument:
            Specifies prefix to be applied to queryband string.
            Types: str

        suffix:
            Optional Argument:
            Specifies suffix to be applied to queryband string.
            Types: str

    EXAMPLES:
        >>> from teradatasqlalchemy.telemetry import collect_queryband
        # Example 1: Collect queryband for a standalone function.
        @collect_queryband(queryband="CreateContext")
        def create_context(host = None, username ...): ...

        # Example 2: Collect queryband for a class method and use
        #            class attribute to retrive queryband string.
        @collect_queryband(attr="func_name")
        def _execute_query(self, persist=False, volatile=False):...

        # Example 3: Collect queryband for a class method and use
        #            method of same class to retrive queryband string.
        @collect_queryband(method="get_class_specific_queryband")
        def _execute_query(self, persist=False, volatile=False):...
    """
    def qb_decorator(exposed_func):
        # This is needed to preserve the docstring of decorated function.
        @wraps(exposed_func)
        def wrapper(*args, **kwargs):
            qb_str = queryband
            # If queryband string is not provided by client while calling decorator,
            # it can be devised using following ways.
            if not qb_str:
                # Approach 1:
                # Extract queryband from value of argument passed
                # to decorated function/method.
                if arg_name:
                    # Extract value from Keyword arguments.
                    if arg_name in kwargs:
                        qb_str = kwargs[arg_name]

                    # Extract value from positional arguments.
                    # Also consider default values.
                    else:
                        # Generate a dictionary containing mapping between
                        # argument names and their run time values.
                        signature = inspect.signature(exposed_func)
                        bound_args = signature.bind(*args, **kwargs)
                        bound_args.apply_defaults()

                        qb_str = bound_args.arguments[arg_name]

                # Approach 2:
                # Extract queryband from an attribute/method associated
                # with class object.
                is_instance_method = args and ('.' in exposed_func.__qualname__)
                if is_instance_method:
                    try:
                        if attr:
                            qb_str = getattr(args[0], attr)
                        elif method:
                            qb_str = getattr(args[0], method)()
                    except Exception as stat_method_err:
                        log("Failed to collect queryband for static class method.", stat_method_err)
                        return exposed_func(*args, **kwargs)
                else:
                    log("Failed to collect queryband for standalone function.")
                    return exposed_func(*args, **kwargs)

            if qb_str:
                # Validate queryband for string type.
                if not isinstance(qb_str, str):
                    log("Failed to collect queryband. Queryband must be of type str not {}".format(type(qb_str)))
                    return exposed_func(*args, **kwargs)

                # Process suffix and prefix.
                if suffix and isinstance(suffix, str):
                    qb_str = qb_str + "_" + suffix
                if prefix and isinstance(prefix, str):
                    qb_str = prefix + "_" + qb_str

                # Validate queryband for allowed characters.
                if not re.match(session_queryband.qb_regex, qb_str):
                    log("Failed to collect queryband. Queryband string: '{}' contains invalid characters. Allowed characters are [a-z, A-Z, 0-9, '_', '-']".format(qb_str))
                    return exposed_func(*args, **kwargs)

                # Append queryband to buffer.
                session_queryband.append_qb(qb_str)

            return exposed_func(*args, **kwargs)

        return wrapper
    return qb_decorator


def set_queryband(con_obj):
    """
    DESCRIPTION:
        Decorator for executing set queryband SQL request using connection object from application
        and then clearing queryband buffer for next workflow.

    PARAMETERS:
        con_obj:
            Required Argument:
            Specifies connection object to execute string.
            Types: Sqlalchemy connection


    EXAMPLES:
        Setting queryband before execution of application's SQL request.
        >>> from teradatasqlalchemy.telemetry import set_queryband
        @set_queryband(con_obj=get_connection())
        def _execute_ddl_statement(ddl_statement):...
    """
    def qb_decorator(execute_func):
        def wrapper(*args, **kwargs):
            # Execute set queryband SQL request.
            try:
                con_obj.exec_driver_sql(session_queryband.generate_set_queryband_query())
            except Exception as qb_err:
                log("Failed to set QB!!!", qb_err)
            # Execute application's SQL request and after successful execution
            # clean queryband buffer.
            try:
                ret_val = execute_func(*args, **kwargs)
            except Exception as exec_err:
                raise
            else:
                session_queryband.reset_qb()
            return ret_val
        return wrapper
    return qb_decorator


def get_qb_query():
    """
    DESCRIPTION:
        Returns a SET queryband SQL query to be used while setting transaction level
        queryband for an application using _QueryBand object.

    PARAMETERS:
        None

    RETURNS:
        str

    RAISES:
        None.

    EXAMPLES:
        >>> from teradatasqlalchemy.telemetry.queryband import get_qb_query
        >>> set_qb_query = get_qb_query()
    """
    return session_queryband.generate_set_queryband_query()


def log(*args):
    """
    DESCRIPTION:
        Prints error message on console if configuration option is enabled.

    PARAMETERS:
        Variable number of arguments each of string type.

    RETURNS:
        None

    EXAMPLES:
        >>> from teradatasqlalchemy.telemetry.queryband import log
        >>> err_str = "SOME_ERR_IN_QUERYBAND"
        >>> log("Failed to collect queryband.", err_str)
    """
    try:
        if session_queryband.verbose:
            print(*args)
    except Exception as log_err:
        if session_queryband.verbose:
            print("Failed to log error in queryband:", log_err)
