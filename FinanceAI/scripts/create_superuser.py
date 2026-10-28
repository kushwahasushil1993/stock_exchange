"""
Bootstrap the first admin account (or promote an existing one).

There's deliberately no HTTP endpoint for this — granting admin access over
the network, even to a "first user", is a bad default. Run it against the
running app's own database:

    docker compose exec api python scripts/create_superuser.py \\
        --email admin@financeai.io --password changeme

    # promote a user who already registered via the UI:
    docker compose exec api python scripts/create_superuser.py \\
        --email someone@example.com --promote-only
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select

from src.api.routers.auth import hash_password
from src.database.models import AsyncSessionFactory, User, init_db


async def run(email: str, password: str, full_name: str, promote_only: bool) -> None:
    await init_db()
    async with AsyncSessionFactory() as db:
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if user:
            user.is_superuser = True
            user.is_active = True
            if not promote_only and password:
                user.hashed_password = hash_password(password)
            await db.commit()
            print(f"✅ Promoted existing user to superuser: {email}")
            return

        if promote_only:
            print(f"❌ No existing user with email {email} — omit --promote-only to create one.")
            return

        if not password:
            print("❌ --password is required when creating a new user.")
            return

        user = User(
            email=email,
            hashed_password=hash_password(password),
            full_name=full_name,
            is_superuser=True,
            is_active=True,
        )
        db.add(user)
        await db.commit()
        print(f"✅ Created superuser: {email}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", default=None, help="Required unless --promote-only")
    parser.add_argument("--full-name", default="Admin")
    parser.add_argument("--promote-only", action="store_true",
                         help="Only flip is_superuser on an existing user; don't create/reset password")
    args = parser.parse_args()
    asyncio.run(run(args.email, args.password, args.full_name, args.promote_only))
