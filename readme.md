# Pywpd
implement the WPDV4 class as a pydantic model mapping the [WPD schema](https://github.com/ankitrohatgi/WebPlotDigitizer/blob/master/docs/JSON_format_specification.md)  with additional conveninence methods to faciliate use and storage and sharing of digitized data.

* Read the weplot digitizer tar and json format (from_tar, from_json)
* Make data readily available in python (to_table)
* Save data back into wpd format (may be usefull in the future to provide a standard format to share figure and data at the same time)

## Roadmap
* Collect archive of digitized rheological data and test if this is a good way to collect and share digitized data

## Examples

[link to quickstart notebook](notebooks/pywpd_quickstart.ipynb)