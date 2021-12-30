# EFCAMDAT
NLP research on the EFCAMDAT dataset

This research is under the guidance of [Leshem Choshen](https://github.com/borgr)
Current steps:
1. Parsing the dataset XML, extracting the original sentence and the corrected sentence.
2. Annotating the correction type with [ERRANT](https://github.com/chrisjbryant/errant)
3. Creating an m2 file of the corrections
4. Saving the metadata in a Dataframe.
5. creating CoNLL-U files for the original and corrected sentences using [this model][model] with [UDpipe][UDPipe] library.
6. Convert the m2 and CoNLL-U files to a syntax based (p.o.s.) m2 file using [this library][gec]
7. Add syntax based correction-types to Dataframe.
8. Add visualisations and analysis.






The [EFCAMDAT dataset](https://philarion.mml.cam.ac.uk/)

The EF-Cambridge Open Language Database (EFCAMDAT) is a publicly available resource to facilitate second language research and teaching.
It contains written samples from thousands of adult learners of English as a second language, world wide.

EFCAMDAT currently contains over 83 million words from 1 million assignments written by 174,000 learners, across a wide range of levels (CEFR stages A1-C2).
This text corpus includes information on learner errors, part of speech, and grammatical relationships.
Researchers can search for language patterns using a range of criteria, including learner nationality and level.

The resource is actively developed by the Department of Theoretical and Applied Linguistics at the University of Cambridge in partnership with Education First.


[udpipe]: https://github.com/ufal/udpipe
[model]: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131
[gec]: https://github.com/borgr/GEC_UD_divergences