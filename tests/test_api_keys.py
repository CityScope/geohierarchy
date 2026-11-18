

def test_save_api_key_keeps_other_keys(tmp_path):
    from geohierarchy.api_keys import load_keys, save_api_key

    f = tmp_path / "keys.json"
    save_api_key("a", "1", f)
    save_api_key("b", "2", f)
    assert load_keys([f]) == {"a": "1", "b": "2"}
    assert oct(f.stat().st_mode)[-3:] == "600"
