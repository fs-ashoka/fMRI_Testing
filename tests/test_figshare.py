from fmri_testing.data.figshare import filter_files, parse_figshare_files


def test_figshare_metadata_parsing_and_filtering():
    payload = {
        "files": [
            {"id": 1, "name": "CSI1_TYPED-FITHRF-GLMDENOISE-RR.nii.gz", "size": 123, "download_url": "https://example.org/a"},
            {"id": 2, "name": "CSI2_OTHER.nii.gz", "size": 456, "download_url": "https://example.org/b"},
        ]
    }
    files = parse_figshare_files(payload)
    assert len(files) == 2
    selected = filter_files(files, regex="TYPED-FITHRF-GLMDENOISE-RR", subjects=["CSI1"])
    assert len(selected) == 1
    assert selected[0].name.startswith("CSI1")
