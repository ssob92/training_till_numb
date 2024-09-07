#
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# DataRobot, Inc.
#
# This is proprietary source code of DataRobot, Inc. and its
# affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, TYPE_CHECKING

import trafaret as t

from datarobot._compat import Int, String
from datarobot.models.api_object import APIObject

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    # We are using the "non-inheritance" instantiation because if trying
    # `class WordCloudNgram(TypedDict):` we have a syntax error thanks to the use of the reserved
    # word `class` - this "namedtuple-like" approach to instantiation gets around that issue
    WordCloudNgram = TypedDict(
        "WordCloudNgram",
        {
            "ngram": str,
            "coefficient": float,
            "count": int,
            "frequency": float,
            "is_stopword": bool,
            "class": Optional[str],
            "variable": Optional[str],
        },
    )


class WordCloud(APIObject):
    """ Word cloud data for the model.

    Notes
    -----
    ``WordCloudNgram`` is a dict containing the following:

        * ``ngram`` (str) Word or ngram value.
        * ``coefficient`` (float) Value from [-1.0, 1.0] range, describes effect of this ngram on \
          the target. Large negative value means strong effect toward negative class in \
          classification and smaller target value in regression models. Large positive - toward \
          positive class and bigger value respectively.
        * ``count`` (int) Number of rows in the training sample where this ngram appears.
        * ``frequency`` (float) Value from (0.0, 1.0] range, relative frequency of given ngram to \
          most frequent ngram.
        * ``is_stopword`` (bool) True for ngrams that DataRobot evaluates as stopwords.
        * ``class`` (str or None) For classification - values of the target class for
          corresponding word or ngram. For regression - None.

    Attributes
    ----------
    ngrams : list of dicts
        List of dicts with schema described as ``WordCloudNgram`` above.
    """

    _converter = t.Dict(
        {
            t.Key("ngrams"): t.List(
                t.Dict(
                    {
                        t.Key("ngram"): String,
                        t.Key("coefficient"): t.Float(gte=-1, lte=1),
                        t.Key("count"): Int,
                        t.Key("frequency"): t.Float(gt=0, lte=1),
                        t.Key("is_stopword"): t.Bool,
                        # Making these optional will allow working with older backends
                        t.Key("class", optional=True, default=None): t.Or(t.String, t.Null),
                        t.Key("variable", optional=True, default=None): t.Or(t.String, t.Null),
                    }
                ).ignore_extra("*")
            )
        }
    ).ignore_extra("*")

    def __init__(self, ngrams: List[WordCloudNgram]) -> None:
        self.ngrams = ngrams

    def __repr__(self) -> str:
        return f"WordCloud({len(self.ngrams)} ngrams)"

    def most_frequent(self, top_n: Optional[int] = 5) -> List[WordCloudNgram]:
        """Return most frequent ngrams in the word cloud.

        Parameters
        ----------
        top_n : int
            Number of ngrams to return

        Returns
        -------
        list of dict
            Up to top_n top most frequent ngrams in the word cloud.
            If top_n bigger then total number of ngrams in word cloud - return all sorted by
            frequency in descending order.
        """
        return sorted(self.ngrams, key=lambda ngram: ngram["frequency"], reverse=True)[:top_n]

    def most_important(self, top_n: Optional[int] = 5) -> List[WordCloudNgram]:
        """Return most important ngrams in the word cloud.

        Parameters
        ----------
        top_n : int
            Number of ngrams to return

        Returns
        -------
        list of dict
            Up to top_n top most important ngrams in the word cloud.
            If top_n bigger then total number of ngrams in word cloud - return all sorted by
            absolute coefficient value in descending order.
        """
        return sorted(self.ngrams, key=lambda ngram: abs(ngram["coefficient"]), reverse=True)[
            :top_n
        ]

    def ngrams_per_class(self) -> Dict[Optional[str], List[WordCloudNgram]]:
        """Split ngrams per target class values. Useful for multiclass models.

        Returns
        -------
        dict
            Dictionary in the format of (class label) -> (list of ngrams for that class)
        """
        per_class = defaultdict(list)
        for ngram in self.ngrams:
            per_class[ngram["class"]].append(ngram)
        return dict(per_class)
