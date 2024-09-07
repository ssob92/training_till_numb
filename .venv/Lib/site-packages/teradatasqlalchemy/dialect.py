# Copyright 2018 by Teradata Corporation. All rights reserved.

from itertools import groupby
from sqlalchemy import pool, String, Numeric
from sqlalchemy import Table, Column, Index
from sqlalchemy.engine import default
from sqlalchemy.sql import select, and_, or_, elements
from sqlalchemy.sql.expression import text, table, column, asc, bindparam
from teradatasqlalchemy.base import TeradataIdentifierPreparer, TeradataExecutionContext
from teradatasqlalchemy.resolver import TeradataTypeResolver
from teradatasqlalchemy import TIMESTAMP, DATE
from teradatasqlalchemy.options import configure

from sqlalchemy import exc
from sqlalchemy.sql import compiler

import sqlalchemy.types as sqltypes
import teradatasqlalchemy.types as tdtypes
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql.functions import func

import json
import re
import datetime

# ischema names is used for reflecting columns (see get_columns in the dialect)
ischema_names = {

    # SQL standard types (modified only to extend _TDComparable)
    'I': tdtypes.INTEGER,
    'I2': tdtypes.SMALLINT,
    'I8': tdtypes.BIGINT,
    'D': tdtypes.DECIMAL,
    'DA': tdtypes.DATE,

    # Numeric types
    'I1': tdtypes.BYTEINT,
    'F': tdtypes.FLOAT,
    'N': tdtypes.NUMBER,

    # Character types
    'CF': tdtypes.CHAR,
    'CV': tdtypes.VARCHAR,
    'CO': tdtypes.CLOB,

    # Datetime types
    'TS': tdtypes.TIMESTAMP,
    'SZ': tdtypes.TIMESTAMP,  # Timestamp with timezone
    'AT': tdtypes.TIME,
    'TZ': tdtypes.TIME,  # Time with timezone

    # Binary types
    'BF': tdtypes.BYTE,
    'BV': tdtypes.VARBYTE,
    'BO': tdtypes.BLOB,

    # Interval types
    'DH': tdtypes.INTERVAL_DAY_TO_HOUR,
    'DM': tdtypes.INTERVAL_DAY_TO_MINUTE,
    'DS': tdtypes.INTERVAL_DAY_TO_SECOND,
    'DY': tdtypes.INTERVAL_DAY,
    'HM': tdtypes.INTERVAL_HOUR_TO_MINUTE,
    'HR': tdtypes.INTERVAL_HOUR,
    'HS': tdtypes.INTERVAL_HOUR_TO_SECOND,
    'MI': tdtypes.INTERVAL_MINUTE,
    'MO': tdtypes.INTERVAL_MONTH,
    'MS': tdtypes.INTERVAL_MINUTE_TO_SECOND,
    'SC': tdtypes.INTERVAL_SECOND,
    'YM': tdtypes.INTERVAL_YEAR_TO_MONTH,
    'YR': tdtypes.INTERVAL_YEAR,

    # Period types
    'PD': tdtypes.PERIOD_DATE,
    'PT': tdtypes.PERIOD_TIME,
    'PZ': tdtypes.PERIOD_TIME,
    'PS': tdtypes.PERIOD_TIMESTAMP,
    'PM': tdtypes.PERIOD_TIMESTAMP,

    # XML type
    'XM': tdtypes.XML,

    # JSON type
    'JN': tdtypes.JSON,

    # UDT
    'UT': tdtypes.TDUDT

}

stringtypes = [t for t in ischema_names if issubclass(ischema_names[t], sqltypes.String)]
# JSON data type in Teradata also has CharacterSet argument.
stringtypes.append('JN')


class TeradataCompiler(compiler.SQLCompiler):

    def __init__(self, dialect, statement, **kwargs):
        super(TeradataCompiler, self).__init__(dialect=dialect,
                                               statement=statement,
                                               **kwargs)

    def get_select_precolumns(self, select, **kwargs):
        """
        handles the part of the select statement before the columns are specified.
        Note: Teradata does not allow a 'distinct' to be specified when 'top' is
              used in the same select statement.

              Instead if a user specifies both in the same select clause,
              the DISTINCT will be used with a ROW_NUMBER OVER(ORDER BY) subquery.
        """

        pre = select._distinct and "DISTINCT " or ""

        # TODO: decide whether we can replace this with the recipe...
        if (select._limit is not None and select._offset is None):
            pre += "TOP %d " % (select._limit)

        return pre

    def visit_mod_binary(self, binary, operator, **kw):
        return self.process(binary.left, **kw) + " MOD " + \
               self.process(binary.right, **kw)

    def visit_ne_binary(self, binary, operator, **kw):
        return self.process(binary.left, **kw) + " <> " + \
               self.process(binary.right, **kw)

    def limit_clause(self, select, **kwargs):
        """Limit after SELECT is implemented in get_select_precolumns"""
        return ""

    def visit_truediv_binary(self, binary, operator, **kw):
        return (
            self.process(binary.left, **kw)
            + " / "
            # TODO: would need a fast cast again here,
            # unless we want to use an implicit cast like "+ 0.0"
            + self.process(
                elements.Cast(
                    binary.right,
                    binary.right.type
                    if binary.right.type._type_affinity is sqltypes.Numeric
                    else tdtypes.NUMBER(),
                ),
                **kw,
            )
        )

    def visit_floordiv_binary(self, binary, operator, **kw):
        return "FLOOR(%s)" % (
                self.process(binary.left, **kw)
                + " / "
                + self.process(binary.right, **kw)
            )


class TeradataDDLCompiler(compiler.DDLCompiler):

    def visit_create_index(self, create, include_schema=False,
                           include_table_schema=True):
        index = create.element
        self._verify_index_table(index)
        preparer = self.preparer
        text = "CREATE "
        if index.unique:
            text += "UNIQUE "
        text += "INDEX %s (%s) ON %s" \
                % (
                    self._prepared_index_name(index,
                                              include_schema=include_schema),
                    ', '.join(
                        self.sql_compiler.process(
                            expr, include_table=False, literal_binds=True) for
                        expr in index.expressions),
                    preparer.format_table(index.table,
                                          use_schema=include_table_schema)
                )
        return text

    def create_table_suffix(self, table):
        """
        This hook processes the optional keyword teradata_suffixes
        ex.
        from teradatasqlalchemy.compiler import\
                        TDCreateTableSuffix as Opts
        t = Table( 'name', meta,
                   ...,
                   teradata_suffixes=Opts.
                                      fallback().
                                      log().
                                      with_journal_table(t2.name)

        CREATE TABLE name, fallback,
        log,
        with journal table = [database/user.]table_name(
          ...
        )

        teradata_suffixes can also be a list of strings to be appended
        in the order given.
        """
        post = table.dialect_kwargs['teradatasql_suffixes']

        if isinstance(post, TDCreateTableSuffix):
            if post.opts:
                return ',\n' + post.compile()
            else:
                return post
        elif post:
            assert type(post) is list
            res = ',\n ' + ',\n'.join(post)
        else:
            return ''

    def post_create_table(self, table):

        """
        This hook processes the TDPostCreateTableOpts given by the
        teradata_post_create dialect kwarg for Table.

        Note that there are other dialect kwargs defined that could possibly
        be processed here.

        See the kwargs defined in dialect.TeradataDialect

        Ex.
        from teradatasqlalchemy.compiler import TDCreateTablePost as post
        Table('t1', meta,
               ...
               ,
               teradata_post_create = post().
                                        fallback().
                                        checksum('on').
                                        mergeblockratio(85)

        creates ddl for a table like so:

        CREATE TABLE "t1" ,
             checksum=on,
             fallback,
             mergeblockratio=85 (
               ...
        )

        """
        kw = table.dialect_kwargs['teradatasql_post_create']
        if isinstance(kw, TDCreateTablePost):
            if kw:
                return '\n' + kw.compile()
        return ''

    def get_column_specification(self, column, **kwargs):

        if column.table is None:
            raise exc.CompileError(
                "Teradata requires Table-bound columns "
                "in order to generate DDL")

        colspec = (self.preparer.format_column(column) + " " + \
                   self.dialect.type_compiler.process(
                       column.type, type_expression=column))

        # Null/NotNull
        if column.nullable is not None:
            if not column.nullable or column.primary_key:
                colspec += " NOT NULL"

        return colspec


class TeradataTypeCompiler(compiler.GenericTypeCompiler):

    def _get(self, key, type_, kw):
        return kw.get(key, getattr(type_, key, None))

    def visit_datetime(self, type_, **kw):
        return self.visit_TIMESTAMP(type_, precision=6, **kw)

    def visit_date(self, type_, **kw):
        return self.visit_DATE(type_, **kw)

    def visit_text(self, type_, **kw):
        return self.visit_CLOB(type_, **kw)

    def visit_time(self, type_, **kw):
        return self.visit_TIME(type_, precision=6, **kw)

    def visit_unicode(self, type_, **kw):
        return self.visit_VARCHAR(type_, charset='UNICODE', **kw)

    def visit_unicode_text(self, type_, **kw):
        return self.visit_CLOB(type_, charset='UNICODE', **kw)

    def visit_boolean(self, type_, **kw):
        return self.visit_BYTEINT(type_, **kw)

    def visit_INTERVAL_YEAR(self, type_, **kw):
        return 'INTERVAL YEAR{}'.format(
            '(' + str(type_.precision) + ')' if type_.precision else '')

    def visit_INTERVAL_YEAR_TO_MONTH(self, type_, **kw):
        return 'INTERVAL YEAR{} TO MONTH'.format(
            '(' + str(type_.precision) + ')' if type_.precision else '')

    def visit_INTERVAL_MONTH(self, type_, **kw):
        return 'INTERVAL MONTH{}'.format(
            '(' + str(type_.precision) + ')' if type_.precision else '')

    def visit_INTERVAL_DAY(self, type_, **kw):
        return 'INTERVAL DAY{}'.format(
            '(' + str(type_.precision) + ')' if type_.precision else '')

    def visit_INTERVAL_DAY_TO_HOUR(self, type_, **kw):
        return 'INTERVAL DAY{} TO HOUR'.format(
            '(' + str(type_.precision) + ')' if type_.precision else '')

    def visit_INTERVAL_DAY_TO_MINUTE(self, type_, **kw):
        return 'INTERVAL DAY{} TO MINUTE'.format(
            '(' + str(type_.precision) + ')' if type_.precision else '')

    def visit_INTERVAL_DAY_TO_SECOND(self, type_, **kw):
        return 'INTERVAL DAY{} TO SECOND{}'.format(
            '(' + str(type_.precision) + ')' if type_.precision else '',
            '(' + str(type_.frac_precision) + ')' if type_.frac_precision is not None else '')

    def visit_INTERVAL_HOUR(self, type_, **kw):
        return 'INTERVAL HOUR{}'.format(
            '(' + str(type_.precision) + ')' if type_.precision else '')

    def visit_INTERVAL_HOUR_TO_MINUTE(self, type_, **kw):
        return 'INTERVAL HOUR{} TO MINUTE'.format(
            '(' + str(type_.precision) + ')' if type_.precision else '')

    def visit_INTERVAL_HOUR_TO_SECOND(self, type_, **kw):
        return 'INTERVAL HOUR{} TO SECOND{}'.format(
            '(' + str(type_.precision) + ')' if type_.precision else '',
            '(' + str(type_.frac_precision) + ')' if type_.frac_precision is not None else '')

    def visit_INTERVAL_MINUTE(self, type_, **kw):
        return 'INTERVAL MINUTE{}'.format(
            '(' + str(type_.precision) + ')' if type_.precision else '')

    def visit_INTERVAL_MINUTE_TO_SECOND(self, type_, **kw):
        return 'INTERVAL MINUTE{} TO SECOND{}'.format(
            '(' + str(type_.precision) + ')' if type_.precision else '',
            '(' + str(type_.frac_precision) + ')' if type_.frac_precision is not None else '')

    def visit_INTERVAL_SECOND(self, type_, **kw):
        if type_.frac_precision is not None and type_.precision:
            return 'INTERVAL SECOND{}'.format(
                '(' + str(type_.precision) + ', ' + str(type_.frac_precision) + ')')
        else:
            return 'INTERVAL SECOND{}'.format(
                '(' + str(type_.precision) + ')' if type_.precision else '')

    def visit_PERIOD_DATE(self, type_, **kw):
        return 'PERIOD(DATE)' + \
               (" FORMAT '" + type_.format + "'" if type_.format is not None else '')

    def visit_PERIOD_TIME(self, type_, **kw):
        return 'PERIOD(TIME{}{})'.format(
            '(' + str(type_.frac_precision) + ')'
            if type_.frac_precision is not None
            else '',
            ' WITH TIME ZONE' if type_.timezone else '') + \
               (" FORMAT '" + type_.format + "'" if type_.format is not None else '')

    def visit_PERIOD_TIMESTAMP(self, type_, **kw):
        return 'PERIOD(TIMESTAMP{}{})'.format(
            '(' + str(type_.frac_precision) + ')'
            if type_.frac_precision is not None
            else '',
            ' WITH TIME ZONE' if type_.timezone else '') + \
               (" FORMAT '" + type_.format + "'" if type_.format is not None else '')

    def visit_TIME(self, type_, **kw):
        tz = ' WITH TIME ZONE' if type_.timezone else ''
        prec = self._get('precision', type_, kw)
        prec = '%s' % '(' + str(prec) + ')' if prec is not None else ''
        return 'TIME{}{}'.format(prec, tz)

    def visit_TIMESTAMP(self, type_, **kw):
        tz = ' WITH TIME ZONE' if type_.timezone else ''
        prec = self._get('precision', type_, kw)
        prec = '%s' % '(' + str(prec) + ')' if prec is not None else ''
        return 'TIMESTAMP{}{}'.format(prec, tz)

    def _string_process(self, type_, datatype, **kw):
        length = self._get('length', type_, kw)
        length = '(%s)' % length if length is not None else ''

        charset = self._get('charset', type_, kw)
        charset = ' CHAR SET %s' % charset if charset is not None else ''

        res = '{}{}{}'.format(datatype, length, charset)
        return res

    def visit_CHAR(self, type_, **kw):
        return self._string_process(type_, 'CHAR', length=type_.length, **kw)

    def visit_VARCHAR(self, type_, **kw):
        if type_.length is None:
            return self._string_process(type_, 'LONG VARCHAR', **kw)
        else:
            return self._string_process(type_, 'VARCHAR', length=type_.length, **kw)

    def visit_CLOB(self, type_, **kw):
        multi = self._get('multiplier', type_, kw)
        if multi is not None and type_.length is not None:
            length = str(type_.length) + multi
            return self._string_process(type_, 'CLOB', length=length, **kw)

        return self._string_process(type_, 'CLOB', **kw)

    def visit_BYTEINT(self, type_, **kw):
        return 'BYTEINT'

    def visit_BYTE(self, type_, **kw):
        return 'BYTE{}'.format(
            '(' + str(type_.length) + ')' if type_.length is not None else '')

    def visit_VARBYTE(self, type_, **kw):
        return 'VARBYTE{}'.format(
            '(' + str(type_.length) + ')' if type_.length is not None else '')

    def visit_BLOB(self, type_, **kw):
        multiplier = self._get('multiplier', type_, kw)
        return 'BLOB{}'.format(
            '(' + str(type_.length) + \
            '{})'.format(multiplier if multiplier is not None else '')
            if type_.length is not None else '')

    def visit_NUMBER(self, type_, **kw):
        args = (str(type_.precision), '') if type_.scale is None \
            else (str(type_.precision), ', ' + str(type_.scale))
        return 'NUMBER{}'.format(
            '' if type_.precision is None else '({}{})'.format(*args))

    def visit_XML(self, type_, **kw):
        return 'XML({0}) INLINE LENGTH {1}'.format(str(type_.maximum_length),
                                                   str(type_.inline_length))

    def visit_JSON(self, type_, **kw):
        sec_part = ''
        # Adding charset if it is not None, storage_format is only being added
        # if it doesn't hold its default value: 'TEXT'. Reason for this being,
        # 'TEXT' is not a valid type name for STORAGE FORMAT.
        if type_.charset is not None:
            sec_part = 'CHARACTER SET {}'.format(type_.charset)
        if type_.storage_format is not None and type_.storage_format != 'TEXT':
            sec_part = '{0} STORAGE FORMAT {1}'.format(sec_part, type_.storage_format)

        return 'JSON({0}) INLINE LENGTH {1} {2}'.format(str(type_.max_length),
                                                        str(type_.inline_length),
                                                        str(sec_part))

    def visit_TDUDT(self, type_, **kw):
        return 'TDUDT{}'.format('' if type_.type_name is None else '(UDTName: {0})'.format(str(type_.type_name)))

    def visit_GEOMETRY(self, type_, **kw):
        return 'ST_GEOMETRY({0}) INLINE LENGTH {1}'.format(str(type_.max_length),
                                                           str(type_.inline_length))

    def visit_MBR(self, type_, **kw):
        return 'MBR'

    def visit_MBB(self, type_, **kw):
        return 'MBB'


class TeradataDialect(default.DefaultDialect):
    name = 'teradatasql'
    driver = 'teradatasql'
    paramstyle = 'qmark'
    default_paramstyle = 'qmark'
    poolclass = pool.SingletonThreadPool

    statement_compiler = TeradataCompiler
    ddl_compiler = TeradataDDLCompiler
    type_compiler = TeradataTypeCompiler
    preparer = TeradataIdentifierPreparer
    execution_ctx_cls = TeradataExecutionContext

    supports_native_boolean = False
    supports_native_decimal = True
    supports_unicode_statements = True
    supports_unicode_binds = True
    postfetch_lastrowid = False
    implicit_returning = False
    preexecute_autoincrement_sequences = False
    case_sensitive = False
    supports_statement_cache = False

    construct_arguments = [
        (Table, {
            "post_create": None,
            "suffixes": None
        }),

        (Index, {
            "order_by": None,
            "loading": None
        }),

        (Column, {
            "compress": None,
            "identity": None
        })
    ]

    def __init__(self, **kwargs):
        super(TeradataDialect, self).__init__(**kwargs)

        # Method to append 'X' at the end of string when "usexviews" is set to 'True'.
        self.__get_xviews_obj = lambda db_obj: db_obj + 'X' if configure.usexviews else db_obj

    def create_connect_args(self, url):

        params = super(TeradataDialect, self).create_connect_args(url)[1]

        if 'username' in params:
            sUserName = params.pop('username')
            if 'user' not in params:  # user URL parameter has higher priority than username prefix before host
                params['user'] = sUserName

        if 'port' in params:
            params['dbs_port'] = str(params['port'])
            del params['port']

        args = json.dumps(params),  # single-element tuple
        kwargs = {}
        return (args, kwargs)

    @classmethod
    def dbapi(cls):

        """ Hook to the dbapi2.0 implementation's module"""
        import teradatasql
        return teradatasql

    @classmethod
    def import_dbapi(cls):

        """ Hook to the dbapi2.0 implementation's module"""
        import teradatasql
        return teradatasql

    def normalize_name(self, name, **kw):
        if name is not None:
            return name.strip()
        return name

    def _is_table_volatile(self, connection, table_name, schema=None):
        """ Internal function to check if the table is a volatile table or not """

        isVolatile = False
        # Volatile tables are always created in users login space.
        # This means that the schema either has to be None or the same as the login user space (the login user name).
        if schema is not None and schema.lower() != self._get_login_user_space(connection).lower():
            return isVolatile

        res = self._get_volatile_tables_list(connection, table_name)
        for r in res:
            if r.lower() == table_name.lower():
                isVolatile = True
                break

        return isVolatile

    def _get_volatile_tables_list(self, connection, table_name=None):
        """ Internal function to get a list of all volatile tables in the current session """
        # TODO: The check can be made convenient by allowing the option to check for only a single table when the
        #       command is modified to 'help volatile table <tablename>' which is currently failing.

        stmt = 'help volatile table'
        res = connection.execute(text(stmt))

        return [row['Table Dictionary Name'] for row in res.mappings()]

    def has_table(self, connection, table_name, schema=None, **kw):
        """
        DESCRIPTION:
            Function to check for the presence of a table in the schema provided, if any.
            Note:
                By default presence of a table is checked irrespective of whether
                user has access to the table or not. This provides faster lookup.
                When option 'configure.usexviews' is set to True, then search for
                the table happens in only user accessible tables.

        PARAMETERS:
            connection:
                Required argument.
                A SQLAlchemy connection object.

            table_name:
                Required argument.
                The name of the table to search for.

            schema:
                Optional argument.
                The schema to search the table in.
                By default, the default schema will be searched.

        RETURNS:
            A Boolean value indicating whether a table named table_name is present or not.

        RAISES:
            None.

        EXAMPLES:
            # Example 1 - Check table 'mytable' exists.
            >>> table_exists = has_table(conn, 'mytable', schema='myschema')

            # Example 2 - Check if table 'mytable' exists and user has access.
            >>> from teradatasqlalchemy.options import configure
            >>> configure.usexviews=True
            >>> table_exists = has_table(conn, 'mytable', schema='myschema')

        """
        schema_name = self.default_schema_name if schema is None else schema

        # Default value for 'usexviews' is False so use dbc.tablesV by default
        # which is faster.
        dbc_tables = self.__get_xviews_obj("tablesV")

        # Permanent tables (TableKind in 'O', 'Q', 'T')
        table_obj = table(dbc_tables, column('DatabaseName'), column('TableName'),
                          column('TableKind'), schema='dbc')
        stmt = select(table_obj.c.TableName) \
            .select_from(table_obj) \
            .where(and_(text('DatabaseName (NOT CASESPECIFIC) = ' + ':schema' + ' (NOT CASESPECIFIC)'),
                        text('TableName=:table_name'),
                        text("TableKind IN ('O', 'Q', 'T')")))
        stmt = text(str(stmt))
        stmt = stmt.bindparams(schema=schema_name, table_name=table_name)
        res = connection.execute(stmt).fetchone()
        table_present = res is not None

        # Volatile tables
        if not table_present:
            table_present = self._is_table_volatile(connection, table_name, schema)

        return table_present

    def has_view(self, connection, view_name, schema=None):
        """
        DESCRIPTION:
            Function to check for the presence of a view in the schema provided, if any.
            Note:
                By default presence of a view is checked irrespective of whether
                user has access to the view or not. This provides faster lookup.
                When option 'configure.usexviews' is set to True, then search for
                the view happens in only user accessible views.

        PARAMETERS:
            connection:
                Required argument.
                A SQLAlchemy connection object.

            view_name:
                Required argument.
                The name of the view to search for.

            schema:
                Optional argument.
                The schema to search the view in.
                By default, the default schema will be searched.

        RETURNS:
            A Boolean value indicating whether a view named view_name is present or not.

        RAISES:
            None.

        EXAMPLES:
            # Example 1 - Check view 'myview' exists.
            >>> view_exists = has_view(conn, 'myview', schema='myschema')

            # Example 2 - Check if view 'myview' exists and user has access.
            >>> from teradatasqlalchemy.options import configure
            >>> usexviews=True
            >>> view_exists = has_view(conn, 'myview', schema='myschema')
        """
        schema_name = self.default_schema_name if schema is None else schema

        # Default value for 'usexviews' is False so use dbc.tablesV by default
        # which is faster.
        dbc_tables = self.__get_xviews_obj("tablesV")

        table_obj = table(dbc_tables, column('DatabaseName'), column('TableName'),
                          column('TableKind'), schema='dbc')
        # Views (TableKind = 'V')
        stmt = select(table_obj.c.TableName) \
            .select_from(table_obj) \
            .where(and_(text('DatabaseName (NOT CASESPECIFIC) = ' + ':schema' + ' (NOT CASESPECIFIC)'),
                        text('TableName=:view_name'),
                        text("TableKind (NOT CASESPECIFIC) = 'V' (NOT CASESPECIFIC)")))

        stmt = text(str(stmt))
        stmt = stmt.bindparams(schema=schema_name, view_name=view_name)
        res = connection.execute(stmt).fetchone()
        view_present = res is not None

        return view_present

    def _resolve_udt_type(self, **kw):
        """
        Resolves the types for UDT columns.
        """
        col_udt = kw["col_udt_name"]
        if col_udt == 'ST_GEOMETRY':
            return tdtypes.GEOMETRY
        elif col_udt == 'MBR':
            return tdtypes.MBR
        elif col_udt == 'MBB':
            return tdtypes.MBB
        else:
            return tdtypes.TDUDT

    def _resolve_type(self, t, **kw):
        """
        Resolves the types for String, Numeric, Date/Time, etc. columns.
        """
        tc = self.normalize_name(t)
        if tc in ischema_names:
            if tc == 'UT':
                type_ = self._resolve_udt_type(**kw)
            else:
                type_ = ischema_names[tc]
            return TeradataTypeResolver().process(type_, typecode=tc, **kw)

        return sqltypes.NullType

    def _get_column_info(self, row):
        """
        Resolves the column information for get_columns given a row.
        """
        chartype = {
            0: None,
            1: 'LATIN',
            2: 'UNICODE',
            3: 'KANJISJIS',
            4: 'GRAPHIC'
        }

        # Handle unspecified characterset and disregard chartypes specified for
        # non-character types (e.g. binary)
        character_set = row['CharType'] if self.normalize_name(row['ColumnType']) in stringtypes else 0

        try:
            inline_length = row['InlineLength']
        except KeyError:
            inline_length = 0

        try:
            storage_format = row['StorageFormat']
        except KeyError:
            storage_format = ""

        try:
            col_udt_name = row['ColumnUDTName']
        except KeyError:
            col_udt_name = ""

        typ = self._resolve_type(row['ColumnType'],
                                 length=int(row['ColumnLength'] or 0),
                                 chartype=chartype[character_set],
                                 prec=int(row['DecimalTotalDigits'] or 0),
                                 scale=int(row['DecimalFractionalDigits'] or 0),
                                 fmt=row['ColumnFormat'],
                                 inline_length=inline_length,
                                 storage_format=storage_format,
                                 col_udt_name=col_udt_name)

        autoinc = row['IdColType'] in ('GA', 'GD')

        # attrs contains all the attributes queried from DBC.ColumnsV
        attrs = {self.normalize_name(k): row[k] for k in row.keys()}
        col_info = {
            'name': self.normalize_name(row['ColumnName']),
            'type': typ,
            'nullable': row['Nullable'] == u'Y',
            'default': row['DefaultValue'],
            'autoincrement': autoinc
        }

        return dict(attrs, **col_info)

    def get_columns(self, connection, table_name, schema=None, **kw):

        # Check if table is a volatile table before the schema is set to the default schema.
        isVolatile = self._is_table_volatile(connection, table_name, schema)

        if schema is None:
            schema = self.default_schema_name

        # Using 'help schema.table.*' statements has been considered.
        # The DBC.ColumnsV provides the default value which is not available
        # with the 'help column' commands result.

        # Check if the object is a view
        schema_name = ':schema'
        # Default value for 'usexviews' is False so use dbc.tablesV by default
        dbc_tables = self.__get_xviews_obj("tablesV")
        table_obj = table(dbc_tables, column('DatabaseName'), column('TableName'),
                          column('tablekind'), column('TVMFlavor'), schema='dbc')

        is_view_c = "CASE WHEN {} = 'V' THEN 1 ELSE 0 END as is_view".format(table_obj.c.tablekind)
        is_art_c = "CASE WHEN {} = 'A' THEN 1 ELSE 0 END as is_art_table".format(table_obj.c.TVMFlavor)

        stmt = select(text(is_view_c + ', ' + is_art_c)) \
            .select_from(table_obj) \
            .where(and_(text('DatabaseName (NOT CASESPECIFIC) = ' + schema_name + ' (NOT CASESPECIFIC)'),
                        text('TableName=:table_name')))

        stmt = text(str(stmt))
        stmt = stmt.bindparams(schema=schema, table_name=table_name)
        res = connection.execute(stmt)
        isView, is_art_table = next(res) if res.rowcount > 0 else (False, False)

        if isView or isVolatile:
            # Volatile table definition is not stored in the dictionary.
            # We use the 'help schema.table.*' command instead to get information for all columns.
            # We have to do the same for views since we need the type information
            # which is not available in dbc.ColumnsV.
            res = self._get_column_help(connection, schema, table_name, column_name=None)

            # If this is a view, get types for individual columns (dbc.ColumnsV won't have types for view columns).
            # For a view or a volatile table, we have to set the default values as the 'help' command does not have it.
            col_info_list = []
            for r in res:
                updated_column_info_dict = self._update_column_help_info(r._mapping)
                col_info_list.append(dict(r._mapping, **(updated_column_info_dict)))
            res = col_info_list
        else:
            # Default value for 'usexviews' is False so use dbc.ColumnsV by default
            dbc_columns = self.__get_xviews_obj("ColumnsV")
            # For permanent tables.
            # ORDER BY ColumnId added to make sure the columns are retrieved in order
            # of their appearance in the table. HELP COLUMN already maintains that order.
            table_obj = table(dbc_columns, schema='dbc')
            stmt = select(text('*')) \
                .select_from(table_obj) \
                .where(and_(text('DatabaseName (NOT CASESPECIFIC) = ' + schema_name + ' (NOT CASESPECIFIC)'),
                            text('TableName=:table_name'))) \
                .order_by(text("ColumnId"))

            stmt = text(str(stmt))
            stmt = stmt.bindparams(schema=schema, table_name=table_name)
            res = connection.execute(stmt).fetchall()
            res =[res_row._mapping for res_row in res]

        final_column_info = []
        # For ART Tables, every layer columns appear in output table. Columns which do not belongs to
        # primary layer will have TSColumnType as 'AS'. Also, some columns will be duplicated so remove
        # duplicates.
        if is_art_table:
            _col_names = set()
            for row in res:
                col_info = self._get_column_info(row)
                if col_info['TSColumnType'].strip() == 'AS' or col_info['ColumnName'] in _col_names:
                    continue
                final_column_info.append(col_info)
                _col_names.add(col_info['ColumnName'])
        else:
            # Ignore the non-functional column in a PTI table
            for row in res:
                col_info = self._get_column_info(row)
                if 'TSColumnType' in col_info and col_info['TSColumnType'] is not None:
                    if col_info['ColumnName'] == 'TD_TIMEBUCKET' and col_info['TSColumnType'].strip() == 'TB':
                        continue
                final_column_info.append(col_info)

        return final_column_info

    def _get_default_schema_name(self, connection):
        res = self.normalize_name(
            connection.execute(text('select database')).scalar())
        return res

    def _get_login_user_space(self, connection):
        """
        Internal function to get the current users login user space.
        This is currently used for volatile table checks.
        """
        res = self.normalize_name(
            connection.execute(text('select user')).scalar())
        return res

    def _get_column_help(self, connection, schema, table_name, column_name):
        """
        Internal function to get the help on a column, when provided, else all columns
        for a table provided using the HELP COLUMN command.

        :param connection:  SQLAlchemy connection object.
        :param schema:      Schema name of the table/view to run 'HELP COLUMN' command for.
        :param table_name:  The name of the table/view to run 'HELP COLUMN' command for.
        :param column_name: Optional column name to get information only about the column.

        :return: The result of the HELP COLUMN command:
                 * A list of dictionaries when column_name is not provided.
                 * A dictionary when column_name is provided.
        """
        prepared = preparer(dialect())
        stmt = 'help column ' + prepared.quote(schema) + '.' + prepared.quote(table_name) + '.' \
               + (prepared.quote(column_name) if column_name else '*')

        result_set = connection.execute(text(stmt)).fetchall()
        return result_set[0] if column_name else result_set

    def _update_column_help_info(self, res):
        """
        Internal function to create new fields in the result dictionary to have the field names
        similar to those for tables.
        """
        return {
            'ColumnName': res['Column Dictionary Name'],
            'ColumnType': res['Type'],
            'ColumnLength': res['Max Length'],
            'CharType': res['Char Type'],
            'DecimalTotalDigits': res['Decimal Total Digits'],
            'DecimalFractionalDigits': res['Decimal Fractional Digits'],
            'ColumnFormat': res['Format'],
            'Nullable': res['Nullable'],
            'DefaultValue': None,
            'IdColType': res['IdCol Type'],
            'TSColumnType': res['Time Series Column Type'] if 'Time Series Column Type' in res else None,
            'ColumnUDTName': res['UDT Dictionary Name'] if 'UDT Dictionary Name' in res else None,
            'InlineLength': res['Inline Length'] if 'Inline Length' in res else None
        }

    def get_table_names(self, connection, schema=None, **kw):
        """
        DESCRIPTION:
            Function to get all the available table names in the schema provided, if any.
            Note:
                By default function returns all tables available irrespective of whether
                user has access to the table or not. This provides faster lookup. In
                case user wants to list down all available tables which user has
                access to, then set option 'configure.usexviews' to True.

        PARAMETERS:
            connection:
                Required argument.
                A SQLAlchemy connection object.

            schema:
                Optional argument.
                The schema to search the table in.
                By default, the default schema will be searched.

        RETURNS:
            A list of table names.

        RAISES:
            None.

        EXAMPLES:
            # Example 1 - Get all the available tables.
            >>> tables = get_table_names(conn, schema='myschema')

            # Example 2 - Get all the user accessible tables.
            >>> from teradatasqlalchemy.options import configure
            >>> configure.usexviews=True
            >>> tables = get_table_names(conn, schema='myschema')
        """
        if schema is None:
            schema = self.default_schema_name

        # Default value for 'usexviews' is False so use dbc.tablesV by default
        dbc_tables = self.__get_xviews_obj("tablesV")

        table_obj = table(dbc_tables, column('tablename'), schema='dbc')
        stmt = select(table_obj.c.tablename) \
            .select_from(table_obj) \
            .where(and_(text('DatabaseName (NOT CASESPECIFIC) = ' + ':schema' + ' (NOT CASESPECIFIC)'),
                        or_(text('tablekind=\'T\''),
                            text('tablekind=\'O\''),
                            text('tablekind=\'Q\''))))
        stmt = text(str(stmt))
        stmt = stmt.bindparams(schema=schema)
        res = connection.execute(stmt).fetchall()
        return [self.normalize_name(name._mapping['TableName']) for name in res]

    def get_schema_names(self, connection, **kw):
        """Retrieves names of Databases/Schemas on the system"""

        # Default value for 'usexviews' is False so use dbc.DatabasesV by default
        dbc_schemas = self.__get_xviews_obj("DatabasesV")
        table_obj = table(dbc_schemas, column('DatabaseName'), schema='dbc')
        stmt = select(table_obj.c.DatabaseName) \
            .select_from(table_obj) \
            .order_by(table_obj.c.DatabaseName)
        res = connection.execute(stmt).fetchall()
        return [self.normalize_name(name._mapping['DatabaseName']) for name in res]

    def get_view_definition(self, connection, view_name, schema=None, **kw):

        if schema is None:
            schema = self.default_schema_name

        res = connection.execute(text('show view {}.{}'.format(schema, view_name))).scalar()
        return self.normalize_name(res)

    def get_view_names(self, connection, schema=None, **kw):
        """
        DESCRIPTION:
            Function to get all the available view names in the schema provided, if any.
            Note:
                By default function returns all views available irrespective of whether
                user has access to the view or not. This provides faster lookup. In
                case user wants to list down all available views which user has
                access to, then set option 'configure.usexviews' to True.

        PARAMETERS:
            connection:
                Required argument.
                A SQLAlchemy connection object.

            schema:
                Optional argument.
                The schema to search the view in.
                By default, the default schema will be searched.

        RETURNS:
            A list of view names.

        RAISES:
            None.

        EXAMPLES:
            # Example 1 - Get all the available views.
            >>> views = get_view_names(conn, schema='myschema')

            # Example 2 - Get all the user accessible views.
            >>> from teradatasqlalchemy.options import configure
            >>> configure.usexviews=True
            >>> tables = get_view_names(conn, schema='myschema')
        """
        if schema is None:
            schema = self.default_schema_name

        # Default value for 'usexviews' is False so use dbc.tablesV by default
        # which is faster.
        dbc_tables = self.__get_xviews_obj("tablesV")

        table_obj = table(dbc_tables, column('tablename'), schema='dbc')
        stmt = select(table_obj.c.tablename) \
            .select_from(table_obj) \
            .where(and_(text('DatabaseName (NOT CASESPECIFIC) = ' + ':schema' + ' (NOT CASESPECIFIC)'),
                        text('tablekind=\'V\'')))

        stmt = text(str(stmt))
        stmt = stmt.bindparams(schema=schema)
        res = connection.execute(stmt).fetchall()
        return [self.normalize_name(name._mapping['TableName']) for name in res]

    def get_pk_constraint(self, connection, table_name, schema=None, **kw):
        """
        Override
        TODO: Check if we need PRIMARY Indices or PRIMARY KEY Indices
        TODO: Check for border cases (No PK Indices)
        """

        if schema is None:
            schema = self.default_schema_name

        # Default value for 'usexviews' is False so use dbc.IndicesV by default
        dbc_indices = self.__get_xviews_obj("IndicesV")

        table_obj = table(dbc_indices, column('ColumnName'), column('IndexName'),
                          schema='dbc')
        stmt = select(table_obj.c.ColumnName, table_obj.c.IndexName) \
            .select_from(table_obj) \
            .where(and_(text('DatabaseName (NOT CASESPECIFIC) = ' + ':schema' + ' (NOT CASESPECIFIC)'),
                        text('TableName=:table'), text('IndexType=:indextype'))) \
            .order_by(asc(column('IndexNumber')))
        stmt = text(str(stmt))
        # K for Primary Key.
        stmt = stmt.bindparams(schema=schema, table=table_name, indextype='K')
        res = connection.execute(stmt).fetchall()

        index_columns = list()
        index_name = None

        for index_column in res:
            index_columns.append(self.normalize_name(index_column._mapping['ColumnName']))
            index_name = self.normalize_name(index_column._mapping['IndexName'])  # There should be just one IndexName

        return {
            "constrained_columns": index_columns,
            "name": index_name
        }

    def get_unique_constraints(self, connection, table_name, schema=None, **kw):
        """
        Overrides base class method
        """
        if schema is None:
            schema = self.default_schema_name

        # Default value for 'usexviews' is False so use dbc.IndicesV by default
        dbc_indices = self.__get_xviews_obj("IndicesV")

        table_obj = table(dbc_indices, column('ColumnName'), column('IndexName'), schema='dbc')
        stmt = select(table_obj.c.ColumnName, table_obj.c.IndexName) \
            .select_from(table_obj) \
            .where(and_(text('DatabaseName (NOT CASESPECIFIC) = ' + ':schema' + ' (NOT CASESPECIFIC)'),
                        text('TableName=:table'),
                        text('IndexType=:indextype'))) \
            .order_by(asc(column('IndexName')))
        stmt = text(str(stmt))
        # U for Unique Key.
        stmt = stmt.bindparams(schema=schema, table=table_name, indextype='U')
        res = connection.execute(stmt).fetchall()

        def grouper(fk_row):
            return {
                'name': self.normalize_name(fk_row.IndexName),
            }

        unique_constraints = list()
        for constraint_info, constraint_cols in groupby(res, grouper):
            unique_constraint = {
                'name': self.normalize_name(constraint_info['name']),
                'column_names': list()
            }

            for constraint_col in constraint_cols:
                unique_constraint['column_names'].append(self.normalize_name(constraint_col._mapping['ColumnName']))

            unique_constraints.append(unique_constraint)

        return unique_constraints

    def get_foreign_keys(self, connection, table_name, schema=None, **kw):
        """
        Overrides base class method
        """

        if schema is None:
            schema = self.default_schema_name
        # Default value for 'usexviews' is False so use DBC.All_RI_ChildrenV by default
        dbc_child_parent_table = self.__get_xviews_obj("All_RI_ChildrenV")

        table_obj = table(dbc_child_parent_table, column('IndexID'), column('IndexName'),
                          column('ChildKeyColumn'), column('ParentDB'),
                          column('ParentTable'), column('ParentKeyColumn'),
                          schema='dbc')
        stmt = select(table_obj.c.IndexID, table_obj.c.IndexName,
                      table_obj.c.ChildKeyColumn, table_obj.c.ParentDB,
                      table_obj.c.ParentTable, table_obj.c.ParentKeyColumn) \
            .select_from(table_obj) \
            .where(and_(text('ChildTable = :table'), text('ChildDB = :schema'))) \
            .order_by(asc(column('IndexID')))

        stmt = text(str(stmt))
        stmt = stmt.bindparams(schema=schema, table=table_name)
        res = connection.execute(stmt).fetchall()

        def grouper(fk_row):
            return {
                'name': fk_row.IndexName or fk_row.IndexID,  # ID if IndexName is None
                'schema': fk_row.ParentDB,
                'table': fk_row.ParentTable
            }

        # TODO: Check if there's a better way
        fk_dicts = list()
        for constraint_info, constraint_cols in groupby(res, grouper):
            fk_dict = {
                'name': constraint_info['name'],
                'constrained_columns': list(),
                'referred_table': constraint_info['table'],
                'referred_schema': constraint_info['schema'],
                'referred_columns': list()
            }

            for constraint_col in constraint_cols:
                fk_dict['constrained_columns'].append(self.normalize_name(constraint_col.ChildKeyColumn))
                fk_dict['referred_columns'].append(self.normalize_name(constraint_col.ParentKeyColumn))

            fk_dicts.append(fk_dict)

        return fk_dicts

    def get_indexes(self, connection, table_name, schema=None, **kw):
        """
        Overrides base class method
        """

        if schema is None:
            schema = self.default_schema_name

        indices = list()
        prepared = preparer(dialect())
        stmt = 'help index ' + prepared.quote(schema) + '.' + prepared.quote(table_name)
        try:
            res = connection.execute(text(stmt)).fetchall()
            res_set = [row._mapping for row in res]
            for row in res_set:
                index_dict = {
                    'name': row['Index Dictionary Name'],
                    'column_names': list(),
                    'unique': True if row['Unique?'] == 'Y' else False
                }

                if "," in row['Column Names']:
                    index_cols = row['Column Names'].split(',')
                else:
                    index_cols = [row['Column Names']]
                for index_col in index_cols:
                    if index_col == 'TD_TIMEBUCKET' and 'Timebucket' in row and row['Timebucket'] is not None:
                        continue
                    else:
                        index_dict['column_names'].append(self.normalize_name(index_col))

                if len(index_dict['column_names']) > 0:
                    indices.append(index_dict)
        except OperationalError as err:
            # This is to catch the following error when the object may be a view and not a table, Like:
            # 1. [Error 3720] This view does not contain any complete index columns of the underlying table.
            # 2. [Error 3823] VIEW 'xxx' may not be used for Help Index/Constraint/Statistics, Update, Delete or Insert.
            if ('3720' in str(err) and 'This view does not contain any complete index columns of the underlying table'
                in str(err)) or \
                    ('3823' in str(
                        err) and 'may not be used for Help Index/Constraint/Statistics, Update, Delete or Insert'
                     in str(err)):
                pass
            else:
                raise

        return indices

    def get_transaction_mode(self, connection, **kw):
        """
        Returns the transaction mode set for the current session.
        T = TDBS
        A = ANSI
        """
        # Default value for 'usexviews' is False so use dbc.sessioninfoV by default
        dbc_sessions = self.__get_xviews_obj("sessioninfoV")

        table_obj = table(dbc_sessions, column('transaction_mode'), column('sessionno'), schema='dbc')
        stmt = select(table_obj.c.transaction_mode) \
            .select_from(table_obj) \
            .where(table_obj.c.sessionno == text('SESSION'))
        res = connection.execute(stmt).scalar()
        return res

    def _get_server_version_info(self, connection, **kw):
        """
        Returns the Teradata Database software version.
        """
        table_obj = table('dbcinfov', column('InfoData'), column('InfoKey'), schema='dbc')
        stmt = select(table_obj.c.InfoData) \
            .select_from(table_obj) \
            .where(table_obj.c.InfoKey == 'VERSION')
        res = connection.execute(stmt).scalar()
        return res

    def conn_supports_autocommit(self, connection, **kw):
        """
        Returns True if autocommit is used for this connection (underlying Teradata session)
        else False
        """
        return self.get_transaction_mode(connection) == 'T'

    def _get_database_names(self, connection, schema_name):
        """
        Function to return a list valid of database names for a given sqlalchemy connection.
        """
        table_obj = table('databasesV', column('databasename'), schema='dbc')
        stmt = select(text(str(func.lower(table_obj.c.databasename)) + ' as databasename')) \
            .select_from(table_obj) \
            .where(text('databasename (NOT CASESPECIFIC) = {} (NOT CASESPECIFIC)'.format(':schema_name')))
        stmt = text(str(stmt))
        stmt = stmt.bindparams(schema_name=schema_name)

        res = connection.execute(stmt).fetchall()
        return [name.databasename for name in res]


dialect = TeradataDialect
preparer = TeradataIdentifierPreparer
compiler = TeradataCompiler


class TeradataOptions(object):
    """
    An abstract base class for various schema object options
    """
    _prepare = preparer(dialect())

    def _append(self, opts, val):
        _opts = opts.copy()
        _opts.update(val)
        return _opts

    def compile(self, **kw):
        """
        processes the argument options and returns a string representation
        """
        pass

    def format_cols(self, key, val, **kw):
        """
        key is a string
        val is a list of strings with an optional dict as the last element
            the dict values are appended at the end of the col list
        """
        res = ''
        col_expr = ', '.join([x for x in val if type(x) is str])

        res += key + '( ' + col_expr + ' )'
        if type(val[-1]) is dict:
            # process syntax elements (dict) after cols
            res += ' '.join(val[-1]['post'])
        return res


class TDCreateTableSuffix(TeradataOptions):
    """
    A generative class for Teradata create table options
    specified in teradata_suffixes
    """

    def __init__(self, opts={}, **kw):
        """
        opts is a dictionary that can be pre-populated with key-value pairs
        that may be overidden if the keys conflict with those entered
        in the methods below. See the compile method to see how the dict
        gets processed.
        """
        self.opts = opts

    def compile(self):
        def process_opts(opts):
            return [key if opts[key] is None else '{}={}'. \
                format(key, opts[key]) for key in opts]

        res = ',\n'.join(process_opts(self.opts))
        return res

    def fallback(self, enabled=True):
        res = 'fallback' if enabled else 'no fallback'
        return self.__class__(self._append(self.opts, {res: None}))

    def log(self, enabled=True):
        res = 'log' if enabled else 'no log'
        return self.__class__(self._append(self.opts, {res: None}))

    def with_journal_table(self, tablename=None):
        """
        tablename is the schema.tablename of a table.
        For example, if t1 is a SQLAlchemy:
                with_journal_table(t1.name)
        """
        return self.__class__(self._append(self.opts, \
                                           {'with journal table': tablename}))

    def before_journal(self, prefix='dual'):
        """
        prefix is a string taking vaues of 'no' or 'dual'
        """
        assert prefix in ('no', 'dual')
        res = prefix + ' ' + 'before journal'
        return self.__class__(self._append(self.opts, {res: None}))

    def after_journal(self, prefix='not local'):
        """
        prefix is a string taking vaues of 'no', 'dual', 'local',
        or 'not local'.
        """
        assert prefix in ('no', 'dual', 'local', 'not local')
        res = prefix + ' ' + 'after journal'
        return self.__class__(self._append(self.opts, {res: None}))

    def checksum(self, integrity_checking='default'):
        """
        integrity_checking is a string taking vaues of 'on', 'off',
        or 'default'.
        """
        assert integrity_checking in ('on', 'off', 'default')
        return self.__class__(self._append(self.opts, \
                                           {'checksum': integrity_checking}))

    def freespace(self, percentage=0):
        """
        percentage is an integer taking values from 0 to 75.
        """
        return self.__class__(self._append(self.opts, \
                                           {'freespace': percentage}))

    def no_mergeblockratio(self):
        return self.__class__(self._append(self.opts, \
                                           {'no mergeblockratio': None}))

    def mergeblockratio(self, integer=None):
        """
        integer takes values from 0 to 100 inclusive.
        """
        res = 'default mergeblockratio' if integer is None \
            else 'mergeblockratio'
        return self.__class__(self._append(self.opts, {res: integer}))

    def min_datablocksize(self):
        return self.__class__(self._append(self.opts, \
                                           {'minimum datablocksize': None}))

    def max_datablocksize(self):
        return self.__class__(self._append(self.opts, \
                                           {'maximum datablocksize': None}))

    def datablocksize(self, data_block_size=None):
        """
        data_block_size is an integer specifying the number of bytes
        """
        res = 'datablocksize' if data_block_size is not None \
            else 'default datablocksize'
        return self.__class__(self._append(self.opts, \
                                           {res: data_block_size}))

    def blockcompression(self, opt='default'):
        """
        opt is a string that takes values 'autotemp',
        'default', 'manual', or 'never'
        """
        return self.__class__(self._append(self.opts, \
                                           {'blockcompression': opt}))

    def with_no_isolated_loading(self, concurrent=False):
        res = 'with no ' + \
              ('concurrent ' if concurrent else '') + \
              'isolated loading'
        return self.__class__(self._append(self.opts, {res: None}))

    def with_isolated_loading(self, concurrent=False, opt=None):
        """
        opt is a string that takes values 'all', 'insert', 'none',
        or None
        """
        assert opt in ('all', 'insert', 'none', None)
        for_stmt = ' for ' + opt if opt is not None else ''
        res = 'with ' + \
              ('concurrent ' if concurrent else '') + \
              'isolated loading' + for_stmt
        return self.__class__(self._append(self.opts, {res: None}))


class TDCreateTablePost(TeradataOptions):
    """
    A generative class for building post create table options
    given in the teradata_post_create keyword for Table
    """

    def __init__(self, opts={}, **kw):
        self.opts = opts

    def compile(self, **kw):
        def process(opts):
            return [key.upper() if opts[key] is None \
                        else self.format_cols(key, opts[key], **kw) \
                    for key in opts if key != 'on commit']

        def process_last(opts):
            if 'on commit' in opts:
                return '\n' + 'on commit {} rows'.format(opts['on commit'])
            else:
                return ''

        return ',\n'.join(process(self.opts)) + process_last(self.opts)

    def on_commit(self, option='delete'):
        assert type(option) is str and option.lower() in ('preserve', 'delete')
        return self.__class__(
            self._append(self.opts, {'on commit': option}))

    def no_primary_index(self):
        return self.__class__(self._append(self.opts, {'no primary index': None}))

    def primary_index(self, name=None, unique=False, cols=[]):
        """
        name is a string for the primary index
        if unique is true then unique primary index is specified
        cols is a list of column names
        """
        res = 'unique primary index' if unique else 'primary index'
        res += ' ' + name if name is not None else ''
        return self.__class__(self._append(self.opts, {res: [self._prepare.quote(c) for c in cols if c is not None]}))

    def __validate_timecode_datatype(self, timecode_datatype):
        """
        Internal function to validate timecode_datatype specified when creating a
        Primary Time Index (PTI) table.

        :param timecode_datatype: The timecode_datatype passed to primary_time_index().

        :return: Boolean value indicating whether the argument is valid (True),
         or raise ValueError/TypeError when invalid.
        """
        # Raise a ValueError is argument is None or not specified - it is required
        if timecode_datatype is None:
            raise ValueError("'timecode_datatype' is a required argument and must not be None.")

        valid_timecode_datatypes = [TIMESTAMP, DATE]
        if type(timecode_datatype) not in valid_timecode_datatypes:
            raise TypeError("timecode_datatype must be of one of the following types: {}. Found {}".
                            format(valid_timecode_datatypes,
                                   type(timecode_datatype)))

        # Looks like the value is valid
        return True

    def __validate_timezero_date(self, timezero_date):
        """
        Internal function to validate timezero_date specified when creating a
        Primary Time Index (PTI) table.

        :param timezero_date: The timezero_date passed to primary_time_index().

        :return: Boolean value indicating whether the argument is valid (True),
         or raise ValueError when invalid.
        """
        # Return True is it is not specified or is None since it is optional
        if timezero_date is None:
            return True

        pattern = re.compile(r"^DATE\s+'(.*)'$")
        match = pattern.match(timezero_date)

        err_msg = "Date format must be: DATE 'YYYY-MM-DD'. Found value with incorrect format: {}".format(timezero_date)
        if match is not None:
            try:
                datetime.datetime.strptime(match.group(1), '%Y-%m-%d')
            except ValueError:
                raise ValueError(err_msg)
        else:
            raise ValueError(err_msg)

        # Looks like the value is valid
        return True

    def __validate_columns_list(self, cols):
        """
        Internal function to validate columns list specified when creating a
        Primary Time Index (PTI) table.

        :param cols: The columns list (cols) passed to primary_time_index().

        :return: Validated column list (list of strings).
         Raise a ValueError/TypeError on validation failure.
        """

        err_msg = "'cols' must be a of type str or list of values of type str. Found: {}"
        col_length_err = "Column name cannot be an empty string. Found: '{}'"

        if cols is None:
            return []

        # Single column (string) specified
        if isinstance(cols, str):
            if len(cols) == 0:
                raise ValueError(col_length_err.format(cols))
            return [cols]
        # list
        elif isinstance(cols, list):
            for col in cols:
                # Must be a list of strings only
                if not isinstance(col, str):
                    raise TypeError(err_msg.format(list(map(type, cols))))
                if len(col) == 0:
                    raise ValueError(col_length_err.format(col))

            # all columns in list validated to be strings
            return cols
        # neither a single string, nor a list
        else:
            raise TypeError(err_msg.format(type(cols)))

    def __validate_timebucket_duration(self, timebucket_duration):
        """
        Internal function to validate timeduration_bucket specified when creating a
        Primary Time Index (PTI) table.

        :param timebucket_duration: The timebucket_duration passed to the primary_time_index().

        :return: Boolean value indicating whether the argument is valid (True).
         or raise ValueError when invalid.
        """
        # Return True is it is not specified or is None since it is optional
        if timebucket_duration is None:
            return True

        if len(timebucket_duration) == 0:
            raise ValueError("timebucket_duration cannot be an empty string.")

        valid_timebucket_durations_formal = ['CAL_YEARS', 'CAL_MONTHS', 'WEEKS', 'DAYS', 'HOURS', 'MINUTES', 'SECONDS',
                                             'MILLISECONDS', 'MICROSECONDS']
        valid_timebucket_durations_shorthand = ['cy', 'cyear', 'cyears',
                                                'cm', 'cmonth', 'cmonths',
                                                'cd', 'cday', 'cdays',
                                                'w', 'week', 'weeks',
                                                'd', 'day', 'days',
                                                'h', 'hr', 'hrs', 'hour', 'hours',
                                                'm', 'mins', 'minute', 'minutes',
                                                's', 'sec', 'secs', 'second', 'seconds',
                                                'ms', 'msec', 'msecs', 'millisecond', 'milliseconds',
                                                'us', 'usec', 'usecs', 'microsecond', 'microseconds']

        # Message for error to be raise when n is invalid
        n_err_msg = "'n' must be a positive integer. Found: {}"

        # Check if notation if formal or shorthand (beginning with a digit)
        if timebucket_duration[0].isdigit():
            for short_notation in valid_timebucket_durations_shorthand:
                pattern = re.compile("^([0-9]+){}$".format(short_notation))
                match = pattern.match(timebucket_duration.lower())
                if match is not None:
                    try:
                        n = int(match.group(1))
                        if n < 0:
                            raise ValueError(n_err_msg.format(n))

                        # Looks like the value is valid
                        return True
                    except ValueError:
                        raise ValueError(n_err_msg.format(match.group(1)))
        else:
            for formal_notation in valid_timebucket_durations_formal:
                pattern = re.compile(r"^{}\(([0-9]+)\)$".format(formal_notation))
                match = pattern.match(timebucket_duration.upper())
                if match is not None:
                    try:
                        n = int(match.group(1))
                        if n < 0:
                            raise ValueError(n_err_msg.format(n))

                        # Looks like the value is valid
                        return True
                    except ValueError:
                        raise ValueError(n_err_msg.format(match.group(1)))

        # Match not found
        raise ValueError("Invalid timebucket_duration: {}".format(timebucket_duration))

    def primary_time_index(self,
                           timecode_datatype,
                           name=None,
                           timezero_date=None,
                           timebucket_duration=None,
                           sequenced=None,
                           seq_max=None,
                           cols=[]):
        """
        timecode_datatype:
            Required Argument.
            Reflection of the form of the timestamp data in the time series.
            Permitted values:
                A teradatasqlalchemy type representing either
                * TIMESTAMP(n),
                * TIMESTAMP(n) WITH TIME ZONE, or
                * DATE.

        name:
            Optional Argument.
            A name for the Primary Time Index (PTI).

        timezero_date:
            Optional Argument.
            Specifies the earliest time series data that the PTI table will accept;
            a date that precedes the earliest date in the time series data.
            Value specified must be of the following format: DATE 'YYYY-MM-DD'
            Default Value: DATE '1970-01-01'.

        timebucket_duration:
            Optional Argument.
            Required if cols is not specified or is empty.
            A duration that serves to break up the time continuum in
            the time series data into discrete groups or buckets.
            Specified using the formal form time_unit(n), where n is a positive
            integer, and time_unit can be any of the following:
            CAL_YEARS, CAL_MONTHS, CAL_DAYS, WEEKS, DAYS, HOURS, MINUTES,
            SECONDS, MILLISECONDS, or MICROSECONDS.

        sequenced:
            Optional Argument.
            Specifies whether the time series data readings are unique in time or not.
            * True implies SEQUENCED, meaning more than one reading from the same
              sensor may have the same timestamp.
            * False implies NONSEQUENCED, meaning there is only one sensor reading
              per timestamp.
              This is the default.

        seq_max:
            Optional Argument.
            Specifies the maximum number of sensor data rows that can have the
            same timestamp. Can be used when 'sequenced' is True.
            Accepted range:  1 - 2147483647.
            Default Value: 20000.

        cols:
            Optional Argument.
            Required if timebucket_duration is not specified.
            A list of one or more PTI table column names.
        """
        # Validate timecode_datatype
        self.__validate_timecode_datatype(timecode_datatype)

        # Validate timebucket_duration
        self.__validate_timebucket_duration(timebucket_duration)

        # Validate timezero_date
        self.__validate_timezero_date(timezero_date)

        # Validate sequenced
        if sequenced is not None:
            if not isinstance(sequenced, bool):
                raise TypeError("'sequenced', when specified, must be of type 'bool'. Found type: {}".
                                format(type(sequenced)))

        # Validate seq_max
        if seq_max is not None:
            if not isinstance(seq_max, int) or \
                    (seq_max < 1 or seq_max > 2147483647):
                raise ValueError("'seq_max' must be a positive integer in the range 1 through 2147483647. Found: {}".
                                 format(seq_max))

        # Validate cols
        cols = self.__validate_columns_list(cols)

        if timebucket_duration is None and len(cols) == 0:
            raise SyntaxError("At least one of 'cols' or 'timebucket_duration' must be specified.")

        res = 'PRIMARY TIME INDEX'
        res += ' ' + name if name is not None else ''

        val = [timecode_datatype.compile(dialect())]
        if timezero_date is not None:
            val.append(timezero_date)
        if timebucket_duration is not None:
            val.append(timebucket_duration)
        if len(cols) > 0:
            val.append('COLUMNS({})'.format(','.join([self._prepare.quote(c) for c in cols if c is not None])))
        if sequenced is not None:
            if sequenced:
                val.append('SEQUENCED{}'.format('(' + str(seq_max) + ')' if seq_max is not None else ''))
            else:
                val.append('NONSEQUENCED')

        return self.__class__(self._append(self.opts, {res: val}))

    def primary_amp(self, name=None, cols=[]):

        """
        name is an optional string for the name of the amp index
        cols is a list of column names (strings)
        """
        res = 'primary amp index'
        res += ' ' + name if name is not None else ''
        return self.__class__(self._append(self.opts, {res: [self._prepare.quote(c) for c in cols if c is not None]}))


    def partition_by_col(self, all_but=False, cols={}, rows={}, const=None):

        """
        ex:

        Opts.partition_by_col(cols ={'c1': True, 'c2': False, 'c3': None},
                     rows ={'d1': True, 'd2':False, 'd3': None},
                     const = 1)
        will emit:

        partition by(
          column(
            column(c1) auto compress,
            column(c2) no auto compress,
            column(c3),
            row(d1) auto compress,
            row(d2) no auto compress,
            row(d3))
            add 1
            )

        cols is a dictionary whose key is the column name and value True or False
        specifying AUTO COMPRESS or NO AUTO COMPRESS respectively. The columns
        are stored with COLUMN format.

        rows is a dictionary similar to cols except the ROW format is used

        const is an unsigned BIGINT
        """
        res = 'partition by( column all but' if all_but else \
            'partition by( column'
        c = self._visit_partition_by(cols, rows)
        c += [{'post': (['add %s' % str(const)]
                        if const is not None
                        else []) + [')']}]

        return self.__class__(self._append(self.opts, {res: c}))

    def _visit_partition_by(self, cols, rows):

        if cols:
            c = ['column(' + self._prepare.quote(k) + ') auto compress ' \
                 for k, v in cols.items() if v is True]

            c += ['column(' + self._prepare.quote(k) + ') no auto compress' \
                  for k, v in cols.items() if v is False]

            c += ['column(' + self._prepare.quote(k) + ')' for k, v in cols.items() if v is None]

        if rows:
            c += ['row(' + k + ') auto compress' \
                  for k, v in rows.items() if v is True]

            c += ['row(' + k + ') no auto compress' \
                  for k, v in rows.items() if v is False]

            c += ['row(' + k + ')' for k, v in rows.items() if v is None]

        return c

    def partition_by_col_auto_compress(self, all_but=False, cols={}, \
                                       rows={}, const=None):

        res = 'partition by( column auto compress all but' if all_but else \
            'partition by( column auto compress'
        c = self._visit_partition_by(cols, rows)
        c += [{'post': (['add %s' % str(const)]
                        if const is not None
                        else []) + [')']}]

        return self.__class__(self._append(self.opts, {res: c}))

    def partition_by_col_no_auto_compress(self, all_but=False, cols={}, \
                                          rows={}, const=None):

        res = 'partition by( column no auto compress all but' if all_but else \
            'partition by( column no auto compression'
        c = self._visit_partition_by(cols, rows)
        c += [{'post': (['add %s' % str(const)]
                        if const is not None
                        else []) + [')']}]

        return self.__class__(self._append(self.opts, {res: c}))

    def index(self, index):
        """
        Index is created with dialect specific keywords to
        include loading and ordering syntax elements

        index is a sqlalchemy.sql.schema.Index object.
        """
        return self.__class__(self._append(self.opts, {res: c}))

    def unique_index(self, name=None, cols=[]):
        res = 'unique index ' + (name if name is not None else '')
        return self.__class__(self._append(self.opts, {res: [self._prepare.quote(c) for c in cols if c is not None]}))

# @compiles(Select, 'teradata')
# def compile_select(element, compiler, **kw):
#    """
#    """
#
#    if not getattr(element, '_window_visit', None):
#      if element._limit is not None or element._offset is not None:
#          limit, offset = element._limit, element._offset
#
#          orderby=compiler.process(element._order_by_clause)
#          if orderby:
#            element = element._generate()
#            element._window_visit=True
#            #element._limit = None
#            #element._offset = None  cant set to none...
#
#            # add a ROW NUMBER() OVER(ORDER BY) column
#            element = element.column(sql.literal_column('ROW NUMBER() OVER (ORDER BY %s)' % orderby).label('rownum')).order_by(None)
#
#            # wrap into a subquery
#            limitselect = sql.select([c for c in element.alias().c if c.key != 'rownum'])
#
#            limitselect._window_visit=True
#            limitselect._is_wrapper=True
#
#            if offset is not None:
#              limitselect.append_whereclause(sql.column('rownum') > offset)
#              if limit is not None:
#                  limitselect.append_whereclause(sql.column('rownum') <= (limit + offset))
#            else:
#              limitselect.append_whereclause(sql.column("rownum") <= limit)
#
#            element = limitselect
#
#    kw['iswrapper'] = getattr(element, '_is_wrapper', False)
#    return compiler.visit_select(element, **kw)
