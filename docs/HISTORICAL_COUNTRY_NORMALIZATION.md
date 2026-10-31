# Historical country normalization

Country labels used in the analytical corpus represent the state to which an affiliation belonged in the publication year, rather than mechanically applying present-day borders to historical addresses. The preserved C1 string is never overwritten.

For the former Soviet republics, the corpus provides publication year but not a reliable publication month/day. The analytical rule therefore uses `USSR` through 1991 inclusive and successor-state labels from 1992. Explicit Soviet-era labels (for example `USSR`, `Ukrainian SSR`, `Georgian SSR`, or common SSR abbreviations) map to `USSR` through 1991. A small set of 1992 records still carries a legacy `USSR` label in WoS; for these, the parser uses an explicit successor-state or unambiguous location token in the preserved address (for example Kiev/Ukraine, Vilnius/Lithuania, or Moscow/Russia). If the successor state cannot be supported from the address, the line is left unresolved rather than assigned to the defunct USSR state. This annual cutover avoids pretending to distinguish publications within the dissolution year more precisely than the bibliographic metadata permit.

`Georgia` is treated specially because the same word occurs in U.S. institutional names. It is accepted as the country Georgia only as an address-tail country token; U.S. state/ZIP syntax such as `Atlanta, GA 30332` is classified as USA. The historical WoS tail `Sukhumi, GA.` is treated as Georgia rather than the U.S. state abbreviation and is then passed through the same year-dependent USSR/successor rule.

JET is handled separately: the geographic field retains UK for a genuine JET address at Culham, while the analytical co-country field substitutes the special transnational actor `JET` only for that affiliation.
