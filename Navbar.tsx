"use client";

import Link from "next/link";
import { useState } from "react";

export default function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <nav className="bg-gradient-to-r from-green-600 to-blue-600 text-white shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Link href="/" className="text-xl font-bold">
                Food-Attention Nexus
              </Link>
            </div>
            <div className="hidden md:block">
              <div className="ml-10 flex items-baseline space-x-4">
                <Link href="/" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-green-700">
                  Home
                </Link>
                <Link href="/overview" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-green-700">
                  Project Overview
                </Link>
                <Link href="/findings" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-green-700">
                  Key Findings
                </Link>
                <Link href="/visualizations" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-green-700">
                  Visualizations
                </Link>
                <Link href="/interactive" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-green-700">
                  Interactive
                </Link>
                <Link href="/recommendations" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-green-700">
                  Recommendations
                </Link>
              </div>
            </div>
          </div>
          <div className="-mr-2 flex md:hidden">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-white hover:bg-green-700 focus:outline-none"
            >
              <span className="sr-only">Open main menu</span>
              {!isMenuOpen ? (
                <svg className="block h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              ) : (
                <svg className="block h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {isMenuOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            <Link href="/" className="block px-3 py-2 rounded-md text-base font-medium hover:bg-green-700">
              Home
            </Link>
            <Link href="/overview" className="block px-3 py-2 rounded-md text-base font-medium hover:bg-green-700">
              Project Overview
            </Link>
            <Link href="/findings" className="block px-3 py-2 rounded-md text-base font-medium hover:bg-green-700">
              Key Findings
            </Link>
            <Link href="/visualizations" className="block px-3 py-2 rounded-md text-base font-medium hover:bg-green-700">
              Visualizations
            </Link>
            <Link href="/interactive" className="block px-3 py-2 rounded-md text-base font-medium hover:bg-green-700">
              Interactive
            </Link>
            <Link href="/recommendations" className="block px-3 py-2 rounded-md text-base font-medium hover:bg-green-700">
              Recommendations
            </Link>
          </div>
        </div>
      )}
    </nav>
  );
}
