from db.base import supabase_client


def get_recipients():
    supabase = supabase_client()
    query = (supabase
            .table("newsletter")
            .select("email")
    )
    res = query.execute()
    
    # return appropiate format
    return [item["email"] for item in (res.data or [])]

if __name__ == "__main__":

    recipients = get_recipients()
    print(recipients)